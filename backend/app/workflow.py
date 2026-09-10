from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .ingestion import MedicalDocumentCleaner
from .models import PatientProfile, SQLQueryResult, WorkflowState
from .llm import OpenAIAnswerer
from .retrieval import HybridRetriever, IntentRouter
from .safety import SafetyEngine
from .sql_guard import ReadOnlySQLGuard


NODE_LABELS = {
    "normalize": "清理输入",
    "extract_profile": "抽取问诊要素",
    "intent": "识别咨询意图",
    "risk": "筛查风险信号",
    "retrieve": "检索可信知识",
    "structured_query": "执行结构化查询",
    "draft": "生成可读建议",
    "safety_review": "执行安全审查",
    "finalize": "整理引用与摘要",
}


class WorkflowCancelled(RuntimeError):
    """Internal signal used to stop a streamed run after client disconnect."""


class WorkflowExecutionError(RuntimeError):
    """A compiled workflow failed during execution and must not be replayed."""


class MedGuideWorkflow:
    SAFE_REVIEW_FALLBACK = (
        "为保障安全，这段自动生成内容已停止展示。MedGuide 不能提供诊断、处方或停药建议；"
        "如需判断病情或调整用药，请向执业医生或药师确认。"
    )
    _UNSAFE_OUTPUT_PATTERNS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            # Action-centered Chinese grammar catches directives even when
            # subjects, pronouns, aspect words, or sentence breaks vary.
            r"(?:你|您|患者|本人)\s*(?:(?:已经|已|肯定|确定|明确|显然|基本|一定)\s*){0,3}"
            r"(?:是|患有|患上|得了)\s*[\u4e00-\u9fffA-Za-z0-9]{1,24}",
            r"(?:已经|已|现已)?\s*(?:确认|确定|证实|明确)\s*(?:诊断)?\s*(?:为|是)\s*"
            r"[\u4e00-\u9fffA-Za-z0-9]{1,24}",
            r"(?:^|[，,。！？!；;\n])\s*(?:(?:我|我们|医生|药师)\s*)?"
            r"(?:建议|要求|请|应当|应该|务必|必须)\s*(?:你|您|患者)?\s*"
            r"(?:先|暂时|立即|马上|现在|直接)?\s*"
            r"(?:(?:把|将)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}\s*"
            r"(?:停一下|停掉|停下|停了|停用|停服|暂停|加量|减量|换药)"
            r"|(?:不要|别|不用|不再)?\s*(?:再|继续)?\s*"
            r"(?:吃|服用|口服|使用|注射|外用|停药|停用|停服|暂停使用|暂停服用|"
            r"停止吃|停止服用|停止使用|加量|减量|换药))",
            r"(?:^|[，,。！？!；;\n])\s*(?:你|您|患者)?\s*"
            r"(?:先|暂时|立即|马上|现在|直接)?\s*"
            r"(?:(?:别|不要|不用|不再)\s*(?:再|继续)?\s*"
            r"(?:吃|服用|口服|使用|注射|外用)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}"
            r"|(?:停止?|停掉|暂停|停止吃|停止服用|停止口服|停止使用|停止注射|停止外用|停用|停服)"
            r"\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}"
            r"|(?:把|将)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}\s*"
            r"(?:停一下|停掉|停下|停了|停用|停服|暂停|加量|减量|换药))",
            r"(?:^|[，,。！？!；;\n])\s*每(?:隔)?\s*"
            r"[\d一二两三四五六七八九十百千万半]+\s*(?:小时|分钟|天|日|周)\s*"
            r"(?:服|服用|口服|吃|使用|注射|外用)\s*"
            r"[\d一二两三四五六七八九十百千万半]+\s*(?:片|粒|毫克|mg|ml|毫升)",
            r"(?:每次|每剂|一次)\s*(?:服|服用|口服|吃|使用|注射|外用)?\s*"
            r"[\d一二两三四五六七八九十百千万半]+\s*"
            r"(?:片|粒|毫克|mg|ml|毫升)",
            r"(?:(?:确诊|诊断)(?:为|是)?|(?:判断|认定)(?:为|是(?!否)))\s*[：:]?\s*"
            r"[\u4e00-\u9fffA-Za-z0-9]{1,24}",
            r"(?:你|您|患者|本人)\s*(?:有|患有|患上|得了|就是|属于)\s*[\u4e00-\u9fffA-Za-z0-9]{1,24}",
            r"(?:你|您|患者|本人)\s*(?:得的是|患的是|是)\s*[\u4e00-\u9fffA-Za-z0-9]{1,24}",
            r"(?:这是|此为|符合)\s*[\u4e00-\u9fffA-Za-z0-9]{1,24}"
            r"|(?:这|此|上述|目前)?(?:症状|表现|情况|检查结果)\s*属于\s*"
            r"[\u4e00-\u9fffA-Za-z0-9]{1,24}",
            r"(?:结果|检查结果|报告)\s*(?:显示|提示|表明|证实)\s*[\u4e00-\u9fffA-Za-z0-9]{1,24}",
            r"(?:diagnosed\s+with|you\s+have|patient\s+has)\s+[A-Za-z][A-Za-z -]{1,30}",
            r"(?:处方|用药方案|治疗方案)\s*[：:]\s*(?!(?:不能|无法|不可|不提供|需由|应由))[^。；\n]{1,80}",
            r"(?:给你|为你|直接)\s*开(?:具)?(?:药|处方)",
            r"(?:把|将)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}\s*(?:停了|停掉|停下|停用|停服|停药|停止使用|停止服用|暂停使用|暂停服用)",
            r"(?:建议|可以|请|应当|应该|务必|立即|直接|自行)\s*(?:把|将)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}\s*(?:停了|停掉|停下|停用|停服|停药|停止使用|停止服用|暂停使用|暂停服用)",
            r"(?:建议|可以|请|应当|应该|务必|立即|直接|自行)\s*(?:停止?服用|停用|停服|停药|停止使用|暂停服用|不要再服用|不要服用|加量|减量|换药)",
            r"(?:建议|可以|请|应当|应该|务必|立即|直接|自行)\s*(?:服用|口服|使用|吃|注射|外用)\s*(?!说明书|药品说明)[\u4e00-\u9fffA-Za-z0-9-]{2,40}",
            r"(?<!不)(?:最好|推荐|建议|可以|可|应当|应该|应|需要|必须|务必|请)\s*"
            r"(?:你|您|患者)?\s*(?:先|暂时|立即|马上|现在|直接)?\s*"
            r"(?:停止?(?:吃|服用|口服|使用|注射|外用)?|停药|停用|停服|暂停使用|暂停服用|"
            r"加量|减量|换药|(?:服|服用|口服|吃|使用|注射|外用))\s*"
            r"[\u4e00-\u9fffA-Za-z0-9-]{0,40}",
            r"(?:^|[。！!；;\n])\s*(?:服用|口服|使用|吃|注射|外用)\s*(?!说明书|药品说明)[\u4e00-\u9fffA-Za-z0-9-]{2,40}",
            r"(?:^|[，,。！!；;\n])\s*(?:(?:所以|因此|那就|那么)\s*)?"
            r"(?:建议|可以|请|应当|应该|务必|立即|现在|马上|先|直接|自行|暂时)?\s*"
            r"(?:停药|停用|停服|停止服用|停止使用|暂停服用|加量|减量|换药)"
            r"[\u4e00-\u9fffA-Za-z0-9-]{0,30}(?=$|[，,。！!；;\n]|并|后|吧)",
            r"(?:^|[，,。！!；;\n])\s*(?:建议|可以|请|应当|应该|务必|立即|直接)\s*"
            r"(?:先|暂时|马上|立即)?\s*(?:不|别|不要|不用)\s*"
            r"(?:吃|服用|口服|使用)\s*[\u4e00-\u9fffA-Za-z0-9-]{2,40}",
            r"(?:^|[，,。！!；;\n])\s*(?:不要|别|切勿|勿)\s*(?:再|继续)?\s*"
            r"(?:吃|服用|口服|使用|注射|外用)\s*[\u4e00-\u9fffA-Za-z0-9-]{2,40}",
            r"(?:^|[，,。！!；;\n])\s*(?:不要|别|切勿|勿)\s*(?:再|继续)?\s*"
            r"(?:停药|停用|停服|停止服用|停止使用|暂停服用|加量|减量|换药)",
            r"(?:take|use|stop|increase|decrease)\s+[A-Za-z][A-Za-z -]{1,30}\s*\d+(?:\.\d+)?\s*(?:mg|milligrams?|ml|tablets?)",
            r"(?:服用|口服|使用)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,30}\s*\d+(?:\.\d+)?\s*(?:mg|毫克|克|片|粒|ml|毫升)",
            r"(?:^|[，,。！!；;\n])\s*(?:服|服用|口服|吃|使用|注射|外用)\s*"
            r"[\d一二两三四五六七八九十百千万半]+\s*(?:mg|毫克|克|片|粒|ml|毫升)",
            r"(?:每次|每剂|一次)\s*\d+(?:\.\d+)?\s*(?:mg|毫克|克|片|粒|ml|毫升)",
            r"(?:每日|每天|一天)\s*[\d一二两三四五六七八九十百千万]+\s*(?:次|片|粒|毫克|mg|毫升|ml)",
            r"(?:^|[。！!；;\n])\s*(?:每日|每天|一天)\s*(?:吃|服用|口服|使用)\s*[\d一二两三四五六七八九十百千万半]+\s*(?:次|片|粒|毫克|mg|ml|毫升)",
            r"[\u4e00-\u9fffA-Za-z0-9-]{2,30}\s*(?:每次|每剂|每日|每天|一天)\s*(?:吃|服用|口服|使用)?\s*[\d一二两三四五六七八九十百千万半]+\s*(?:次|片|粒|毫克|mg|ml|毫升)",
            r"[\u4e00-\u9fffA-Za-z0-9-]{2,30}\s*每\s*[\d一二两三四五六七八九十百千万半]+\s*(?:小时|分钟|日|天)\s*(?:服用|口服|使用|吃|服)?\s*[一二两三四五六七八九十百千万\d]+\s*(?:片|粒|毫克|mg|ml|毫升)",
            r"(?:建议|可以|请|应当|应该|务必|立即|直接|自行)\s*(?:把|将)?\s*(?:药物|药品|用药|剂量)\s*(?:加倍|加量|减半|减量|增加|减少|调高|调低)",
            r"(?:一日|每隔)\s*[\d一二两三四五六七八九十百千万]+\s*(?:次|片|粒|毫克|mg|毫升|ml)",
            r"(?:保证|一定|绝对)\s*(?:能|会|治好|有效)",
        )
    )
    _SAFE_NEGATED_OUTPUT = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?:不能|不可|无法|不提供|不会|不得|尚不能|未能|尚未)\s*"
            r"(?:直接|确定性地?)?\s*"
            r"(?:诊断|确诊|判断|确认|确定|证实|断定)(?:为|是)?\s*[：:]?\s*"
            r"(?:(?!但是|然而|不过|可是|却|但|所以|因此|那就|那么|"
            r"建议|请|应当|应该|务必|立即|直接)[\u4e00-\u9fffA-Za-z0-9]){0,24}",
            r"(?:不能|不可|无法|不提供|不会|不得)\s*"
            r"(?:给出|提供|开具|进行|做出)?\s*"
            r"(?:处方|用药方案|治疗方案|停药|停用|停服|加量|减量|换药)\s*"
            r"(?:建议|指令|结论|方案)?",
            r"(?:(?:(?:我|我们|医生|药师)\s*)?(?:建议|提醒|请)\s*(?:你|您|患者)?\s*)?"
            r"(?:不能|不可|不要|请勿|切勿|不应|不得|不建议)\s*(?:自行|随意|擅自)\s*"
            r"(?:停药|停用|停服|停止服用|停止使用|暂停服用|服用|口服|使用|吃药|"
            r"加量|减量|加倍|减半|调高|调低|换药|调整用药)"
            r"(?:(?!但是|然而|不过|可是|却|但|所以|因此|那就|那么|建议|请|"
            r"应当|应该|务必|立即|直接)[\u4e00-\u9fffA-Za-z0-9-]){0,30}",
            r"(?:停药|停用|停服|停止服用|停止使用)(?:后|前|时)\s*"
            r"(?:可能|会|是否|有哪些|出现|发生|需要注意|有什么)"
            r"[^，,。！!；;\n]{0,30}(?:吗|？|\?)",
            r"(?:服用|口服|使用|吃|注射|外用)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}\s*"
            r"(?:有|会有)?\s*(?:哪些|什么)?\s*(?:禁忌症?|副作用|不良反应|注意事项|相互作用)"
            r"[^，,。！!；;\n]{0,20}(?:吗|呢|？|\?)",
            r"(?:服用|口服|使用|吃|注射|外用)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}\s*"
            r"(?:后|时)?\s*(?:可能|有时|偶尔|常见)\s*(?:会)?\s*(?:出现|发生|引起|导致|伴有)?"
            r"(?:(?!建议|推荐|最好|停药|停用|停服|加量|减量|换药)[^，,。！!；;\n]){1,40}",
            r"(?:服用|口服|使用|吃|注射|外用)\s*[\u4e00-\u9fffA-Za-z0-9-]{1,40}\s*"
            r"(?:的)?\s*(?:常见)?\s*(?:禁忌症?|副作用|不良反应|注意事项|相互作用)\s*"
            r"(?:包括|有|为|是)\s*"
            r"(?:(?!建议|推荐|最好|停药|停用|停服|加量|减量|换药)[^，,。！!；;\n]){1,40}",
            r"(?:不能|不可|无法|不|未必)\s*(?:保证|一定|绝对)\s*(?:能|会|治好|有效)",
        )
    )

    def __init__(
        self,
        retriever: HybridRetriever,
        sql_guard: ReadOnlySQLGuard,
        safety: SafetyEngine,
        answerer: OpenAIAnswerer | None = None,
    ) -> None:
        self.retriever = retriever
        self.sql_guard = sql_guard
        self.safety = safety
        self.router = IntentRouter()
        self.answerer = answerer or OpenAIAnswerer()
        self.cleaner = MedicalDocumentCleaner()
        self.langgraph_available = self._check_langgraph()
        self.graph = self._build_graph() if self.langgraph_available else None

    @staticmethod
    def _check_langgraph() -> bool:
        try:
            import langgraph  # noqa: F401

            return True
        except ImportError:
            return False

    def _build_graph(self):
        """Compile the same node contract for deployments that install LangGraph."""
        try:
            from langgraph.graph import END, START, StateGraph

            graph = StateGraph(WorkflowState)
            node_methods = {
                "normalize": self.normalize,
                "extract_profile": self.extract_profile,
                "intent": self.intent,
                "risk": self.risk,
                "retrieve": self.retrieve,
                "structured_query": self.structured_query,
                "draft": self.draft,
                "safety_review": self.safety_review,
                "finalize": self.finalize,
            }
            for name, method in node_methods.items():
                graph.add_node(name, method)
            graph.add_edge(START, "normalize")
            ordered = tuple(node_methods)
            for current, following in zip(ordered, ordered[1:]):
                graph.add_edge(current, following)
            graph.add_edge("finalize", END)
            return graph.compile()
        except Exception:
            # Optional dependency or version mismatch should not break offline mode.
            return None

    def run(
        self,
        state: WorkflowState,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: Any | None = None,
    ) -> WorkflowState:
        started = time.perf_counter()
        state = dict(state)
        state["events"] = []
        if self.graph is not None and on_event is None:
            try:
                compiled_result = dict(self.graph.invoke(state))
                compiled_result["events"] = [
                    {"type": "node", "node": name, "label": NODE_LABELS[name], "status": "completed", "elapsed_ms": 0.0}
                    for name in NODE_LABELS
                ]
                compiled_result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                compiled_result["workflow_engine"] = "langgraph"
                return compiled_result
            except Exception as exc:
                # A runtime graph failure may occur after retrieval, SQL, or
                # model calls. Replaying from normalize would duplicate those
                # side effects, so compatibility fallback is compile-time only.
                raise WorkflowExecutionError("问诊工作流执行失败，请稍后重试") from exc
        nodes = (
            self.normalize,
            self.extract_profile,
            self.intent,
            self.risk,
            self.retrieve,
            self.structured_query,
            self.draft,
            self.safety_review,
            self.finalize,
        )
        for node in nodes:
            if cancel_event is not None and cancel_event.is_set():
                raise WorkflowCancelled()
            name = node.__name__
            node_started = time.perf_counter()
            state = node(state)
            event = {
                "type": "node",
                "node": name,
                "label": NODE_LABELS.get(name, name),
                "status": "completed",
                "elapsed_ms": round((time.perf_counter() - node_started) * 1000, 2),
            }
            state.setdefault("events", []).append(event)
            if on_event:
                on_event(event)
        if cancel_event is not None and cancel_event.is_set():
            raise WorkflowCancelled()
        state["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        # ``langgraph_available`` describes an installed capability, not the
        # runner used for this invocation.  SSE/cancellation uses the
        # deterministic loop so its event and cancellation semantics are
        # explicit.
        state["workflow_engine"] = "deterministic"
        return state

    def normalize(self, state: WorkflowState) -> WorkflowState:
        message = state.get("user_message", "").strip()
        state["normalized_message"] = re.sub(r"\s+", " ", message)
        return state

    _PROFILE_SYMPTOMS = (
        "发热", "咳嗽", "咽痛", "腹痛", "头痛", "胸痛", "皮疹", "头晕", "恶心", "呕吐", "乏力",
        "流鼻涕", "流涕", "鼻塞", "呼吸困难", "气促",
    )
    _PROFILE_SPLIT = re.compile(r"[，,。；;！？!?]|但是|可是|然而|但")
    _PROFILE_THIRD_PARTY = re.compile(
        r"男朋友|女朋友|朋友|家人|父亲|母亲|爸爸|妈妈|儿子|女儿|丈夫|妻子|老公|老婆|同事|邻居|他人|他|她|他们|她们"
    )
    _PROFILE_PATIENT_MARKER = re.compile(r"我|本人|患者|自己|孩子|宝宝|孕妇")
    _PROFILE_NEGATION_BEFORE = re.compile(
        r"(?:没有|没|无|否认|不曾|从未|不是|并非|未见|未|不再|不伴有|不伴随|不存在)"
        r"(?:(?:明显|严重|出现|发生|感到|觉得|有|伴有|伴随|过))*$"
    )
    _PROFILE_NEGATION_AFTER = re.compile(
        r"^(?:(?:的|这个症状)?(?:不是|并非)(?:我|本人|患者|自己)|"
        r"(?:都|均|全部)?(?:没有|无|否认|未(?:见|有|出现|发生)?|不存在|不伴有|不伴随))"
    )
    _PROFILE_KNOWLEDGE = re.compile(
        r"(?:有哪些|有什么|是什么|什么是|的表现|症状包括|如何预防|怎么判断|怎么区分|会不会|是否|有无|有没有)"
    )
    _PROFILE_HELP = re.compile(r"怎么办|需要急诊|需要就医|怎么处理|如何缓解|严重吗|危险吗")
    _DURATION = re.compile(
        r"(?P<value>\d{1,4}|[零〇一二两三四五六七八九十百]+|半)\s*"
        r"(?P<unit>分钟|小时|天|周|个月|月|年)"
    )
    _DURATION_PREFIX = re.compile(r"(?:持续|已经|已有|大约|约|大概|差不多)\s*$")
    _DURATION_NEGATION = re.compile(r"(?:不是|并非|没有|没|否认|不到|不足|不满)\s*$")
    _DURATION_MAXIMUM = {
        "分钟": 10080,
        "小时": 8760,
        "天": 3650,
        "周": 520,
        "个月": 120,
        "月": 120,
        "年": 100,
    }

    @classmethod
    def _profile_clauses(cls, text: str) -> list[str]:
        return [part.strip() for part in cls._PROFILE_SPLIT.split(text) if part.strip()]

    @classmethod
    def _patient_clause(cls, clause: str) -> bool:
        """Exclude symptoms that are explicitly attributed to somebody else."""
        third_party = cls._PROFILE_THIRD_PARTY.search(clause)
        if not third_party:
            return True
        # ``我也胸痛``/``我自己咳嗽`` explicitly re-attaches the clause to the
        # patient; otherwise a relative's symptom must not enter this profile.
        patient_after = clause[third_party.end():]
        return bool(re.search(r"(?:我|本人|自己)\s*(?:也|还|同时|自己|有|出现|感到)", patient_after))

    @classmethod
    def _asserted_profile_term(cls, clause: str, term: str) -> bool:
        for match in re.finditer(re.escape(term), clause):
            before = clause[:match.start()]
            after = clause[match.end():]
            if cls._PROFILE_NEGATION_BEFORE.search(before) or cls._PROFILE_NEGATION_AFTER.match(after):
                continue
            # A negator can scope over a coordinated symptom list (for
            # such as ``没有胸痛或呼吸困难``), or appear after the list
            # (``胸痛和呼吸困难都没有``). Do not let the conjunction make
            # the second term look asserted just because the negator is no
            # longer adjacent to it.
            if re.search(
                r"(?:没有|无|否认|不曾|从未|不是|并非|未见|没|不再|未|不伴有|不伴随|不存在)"
                r"[^，。；;！？!?]{0,20}(?:和|或|及|与|、|以及)$",
                before,
            ):
                continue
            if re.search(
                r"(?:和|或|及|与|、|以及|并且)[^，。；;！？!?]{0,30}"
                r"(?:都|均|全部)?(?:没有|无|否认|未(?:见|有|出现|发生)?|不存在|不伴有|不伴随)$",
                f"{before}{term}{after}",
            ):
                continue
            if cls._PROFILE_KNOWLEDGE.search(clause) and not cls._PROFILE_HELP.search(clause):
                continue
            if ("?" in clause or "？" in clause or "吗" in clause) and not cls._PROFILE_HELP.search(clause):
                continue
            return True
        return False

    @staticmethod
    def _chinese_number(value: str) -> int | None:
        digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        units = {"十": 10, "百": 100}
        total = 0
        current = 0
        for character in value:
            if character in digits:
                current = digits[character]
            elif character in units:
                total += (current or 1) * units[character]
                current = 0
            else:
                return None
        return total + current

    @classmethod
    def _extract_duration(cls, clause: str) -> str | None:
        for match in cls._DURATION.finditer(clause):
            before = clause[:match.start()]
            if cls._DURATION_NEGATION.search(before):
                continue
            explicit = bool(cls._DURATION_PREFIX.search(before))
            symptom_context = any(cls._asserted_profile_term(clause, symptom) for symptom in cls._PROFILE_SYMPTOMS)
            standalone = bool(
                re.fullmatch(
                    rf"(?:大约|约|大概|差不多)?\s*{cls._DURATION.pattern}\s*(?:左右|了|多)?",
                    clause,
                )
            )
            if not (explicit or symptom_context or standalone):
                continue
            raw_value = match.group("value")
            unit = match.group("unit")
            if raw_value == "半":
                return f"半{unit}"
            numeric = int(raw_value) if raw_value.isdigit() else cls._chinese_number(raw_value)
            if numeric is None or numeric <= 0 or numeric > cls._DURATION_MAXIMUM[unit]:
                continue
            return f"{numeric}{unit}"
        return None

    def extract_profile(self, state: WorkflowState) -> WorkflowState:
        profile: PatientProfile = dict(state.get("profile", {}))
        text = state.get("normalized_message", "")
        clauses = self._profile_clauses(text)
        patient_clauses = [clause for clause in clauses if self._patient_clause(clause)]

        # Parse explicit patient phrases before short gender tokens, so
        # ``我男朋友...，我是女性`` cannot be overwritten by ``男``.
        for clause in patient_clauses:
            sex_match = re.search(
                r"(?:我是|本人是|患者是|患者为|我为|自己是)\s*(女性|女士|女|男性|男士|男)", clause
            )
            if sex_match and not re.search(r"(?:不是|并非|没有|无)\s*$", clause[:sex_match.start()]):
                profile["sex"] = "女" if sex_match.group(1) in {"女性", "女士", "女"} else "男"
                break
        if "sex" not in profile:
            for clause in patient_clauses:
                if re.search(r"(?:^|[：:]|患者(?:为|是)?|本人(?:为|是)?|我(?:是|为)?)\s*(?:女性|女士|女)\b", clause):
                    profile["sex"] = "女"
                    break
                if re.search(r"(?:^|[：:]|患者(?:为|是)?|本人(?:为|是)?|我(?:是|为)?)\s*(?:男性|男士|男)\b", clause):
                    profile["sex"] = "男"
                    break

        for clause in patient_clauses:
            age_match = re.search(r"(?<!\d)(\d{1,3})\s*岁", clause)
            if age_match:
                profile["age"] = int(age_match.group(1))
                break
        for clause in patient_clauses:
            duration = self._extract_duration(clause)
            if duration:
                profile["duration"] = duration
                break

        symptoms = list(profile.get("symptoms", []))
        for symptom in self._PROFILE_SYMPTOMS:
            if any(self._asserted_profile_term(clause, symptom) for clause in patient_clauses):
                if symptom not in symptoms:
                    symptoms.append(symptom)
        profile["symptoms"] = symptoms
        if symptoms and not profile.get("chief_complaint"):
            profile["chief_complaint"] = symptoms[0]
        for clause in patient_clauses:
            history_match = re.search(r"(?:既往史|病史|以前有|曾经)[:：]?\s*([^，。；;]+)", clause)
            if history_match:
                profile["history"] = history_match.group(1).strip()
                break
            negative_history_match = re.search(
                r"(?:没有|无|否认)\s*([^，。；;]{1,24}?)(?:等)?(?:既往病史|病史)",
                clause,
            )
            if negative_history_match:
                profile["history"] = f"否认{negative_history_match.group(1).strip()}"
                break
        profile["associated_symptoms"] = [word for word in symptoms if word != profile.get("chief_complaint")]
        state["profile"] = profile
        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        return state

    def intent(self, state: WorkflowState) -> WorkflowState:
        current = self.router.classify(state.get("normalized_message", ""))
        previous = state.get("intent", "unknown")
        if current == "unknown":
            profile_intent = self.router.classify(self.router.rewrite("", state.get("profile", {})))
            message = state.get("normalized_message", "")
            looks_like_new_question = any(
                marker in message
                for marker in ("什么", "怎么", "如何", "是否", "能否", "哪些", "吗", "？", "?")
            )
            if previous in {"disease", "drug", "exam", "department", "faq"} and not looks_like_new_question:
                current = previous
            elif profile_intent != "unknown" and not looks_like_new_question:
                current = profile_intent
        state["intent"] = current
        return state

    def risk(self, state: WorkflowState) -> WorkflowState:
        assessment = self.safety.screen(state.get("normalized_message", ""), state.get("profile", {}))
        previous_level = state.get("risk_level")
        previous_flags = list(state.get("risk_flags", []))
        sticky_high = previous_level == "high"
        if sticky_high or assessment.level == "high":
            state["risk_level"] = "high"
            state["risk_flags"] = list(dict.fromkeys(previous_flags + list(assessment.flags)))
            state["risk_advice"] = (
                "建议立即联系当地急救服务或前往最近的急诊；不要仅依赖线上问答等待观察。"
            )
        elif previous_level == "watch" or assessment.level == "watch":
            state["risk_level"] = "watch"
            state["risk_flags"] = list(dict.fromkeys(previous_flags + list(assessment.flags)))
            state["risk_advice"] = assessment.advice
        else:
            state["risk_level"] = assessment.level
            state["risk_flags"] = list(assessment.flags)
            state["risk_advice"] = assessment.advice
        history = list(state.get("risk_history", []))
        history.append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "level": state["risk_level"],
                "flags": list(state.get("risk_flags", [])),
                "source": "sticky-history" if sticky_high else "rules",
            }
        )
        state["risk_history"] = history[-20:]
        state["risk_assessed"] = True
        return state

    def retrieve(self, state: WorkflowState) -> WorkflowState:
        intent = state.get("intent", "unknown")
        if intent in ("structured",) or state.get("risk_level") == "high":
            state["citations"] = []
            return state
        query = self.router.rewrite(state.get("normalized_message", ""), state.get("profile", {}))
        state["rewritten_query"] = query
        state["citations"] = [citation.as_dict() for citation in self.retriever.search(query, intent, top_k=4)]
        return state

    def structured_query(self, state: WorkflowState) -> WorkflowState:
        if state.get("intent") != "structured":
            state["structured_result"] = None
            return state
        if state.get("risk_level") == "high":
            blocked = SQLQueryResult(
                intent="structured",
                sql="",
                params=(),
                columns=(),
                rows=(),
                blocked=True,
                reason="检测到高风险信号，已优先执行就医分流；暂不执行普通数据查询",
            )
            state["structured_result"] = blocked.as_dict()
            return state
        result = self.sql_guard.from_natural_language(state.get("normalized_message", ""))
        state["structured_result"] = result.as_dict() if result else None
        return state

    def draft(self, state: WorkflowState) -> WorkflowState:
        text = state.get("normalized_message", "")
        profile = state.get("profile", {})
        if state.get("risk_level") == "high":
            flags = "、".join(state.get("risk_flags", []))
            response = f"当前描述包含需要优先排除的风险信号：{flags}。{state.get('risk_advice')}"
            response += " 在等待帮助时尽量保持安全体位，并准备好正在使用的药物和既往病史信息。"
            state["answer_source"] = "safety-rules"
        elif state.get("structured_result"):
            result = state["structured_result"]
            if result.get("blocked"):
                response = f"为了保护数据安全，这次查询没有执行：{result.get('reason')}。可以换成查询药品库存、科室排班或检查价格。"
            elif result.get("rows"):
                response = self._format_rows(result)
            else:
                response = "当前数据源中暂时没有匹配记录；可以补充更具体的药品、科室或检查名称。"
            state["answer_source"] = "read-only-query"
        else:
            citations = state.get("citations", [])
            if citations:
                context = "\n".join(f"[{item['title']}] {item['snippet']}" for item in citations)
                # Minimize identity data at the final outbound boundary.  The
                # stored session keeps the user's original words for the local
                # workflow, while an external provider sees only sanitized
                # query/context text.
                outbound_query = self._external_query_summary(state)
                outbound_context = self.cleaner.sanitize_for_model(context)
                generated = self.answerer.generate(
                    outbound_query,
                    state.get("risk_level", "low"),
                    outbound_context,
                )
                if generated:
                    response = generated
                    state["answer_source"] = "openai-grounded"
                elif getattr(self.answerer, "required", False):
                    raise WorkflowExecutionError("外部生成服务暂时不可用")
                else:
                    lead = citations[0]["snippet"]
                    response = f"根据当前知识库，和你描述最相关的资料是“{citations[0]['title']}”：{lead}"
                    response += "\n\n这只是基于有限信息的健康信息整理，不代表诊断。若症状持续、加重或影响日常活动，建议预约线下医生。"
                    state["answer_source"] = "retrieval-template"
            else:
                response = "我还缺少足够的背景信息来给出可靠的分流建议。请补充年龄、症状持续多久，以及是否伴随发热、疼痛加重或呼吸困难。"
                state["answer_source"] = "clarification-template"
        missing = self._next_question(profile, state.get("risk_level", "low"), text)
        state["next_question"] = missing
        if missing and state.get("risk_level") != "high" and not state.get("structured_result"):
            response += f"\n\n为了缩小分流范围，下一步想确认：{missing}"
        state["response"] = response
        return state

    def _external_query_summary(self, state: WorkflowState) -> str:
        """Build an allowlisted clinical summary; never send raw patient prose."""
        profile = state.get("profile", {})
        parts = [f"咨询类别：{state.get('intent', 'unknown')}"]
        age = profile.get("age")
        if isinstance(age, int) and 0 < age <= 120:
            parts.append(f"年龄：{age}岁")
        if profile.get("sex") in {"男", "女"}:
            parts.append(f"性别：{profile['sex']}")
        complaint = profile.get("chief_complaint")
        if complaint in self._PROFILE_SYMPTOMS:
            parts.append(f"主要症状：{complaint}")
        duration = str(profile.get("duration", ""))
        if duration and re.fullmatch(r"[\d一二两三四五六七八九十半]+(?:分钟|小时|天|周|个月|月|年)", duration):
            parts.append(f"持续时间：{duration}")
        associated = profile.get("associated_symptoms")
        if isinstance(associated, list):
            safe_associated = [item for item in associated if item in self._PROFILE_SYMPTOMS]
            if safe_associated:
                parts.append("伴随症状：" + "、".join(safe_associated[:6]))
        flags = [str(flag) for flag in state.get("risk_flags", []) if str(flag).strip()]
        if flags:
            parts.append("风险信号：" + "、".join(flags[:6]))
        titles = [
            self.cleaner.sanitize_for_model(str(item.get("title", "")))
            for item in state.get("citations", [])
            if isinstance(item, dict) and item.get("title")
        ]
        if titles:
            parts.append("资料主题：" + "、".join(titles[:4]))
        return "；".join(parts)

    @staticmethod
    def _next_question(profile: PatientProfile, risk_level: str, text: str) -> str | None:
        if risk_level == "high":
            return None
        if not profile.get("age"):
            return "患者大致年龄是多少？"
        if not profile.get("duration"):
            return "症状从什么时候开始，持续了多久？"
        if not profile.get("associated_symptoms") and any(word in text for word in ("发热", "咳嗽", "腹痛", "头痛", "皮疹")):
            return "是否伴随发热、明显疼痛加重、呼吸困难或其他新出现的症状？"
        return None

    @staticmethod
    def _format_rows(result: dict[str, Any]) -> str:
        rows = result.get("rows", [])
        lines = ["已通过只读白名单查询到以下数据："]
        for row in rows[:5]:
            lines.append(" · " + " / ".join(f"{key}: {value}" for key, value in row.items()))
        return "\n".join(lines) + "\n\n结果来自当前已连接的数据源；实际库存、排班和价格请以医院实时系统为准。"

    def safety_review(self, state: WorkflowState) -> WorkflowState:
        response = unicodedata.normalize("NFKC", str(state.get("response", "")))
        response = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", response)
        compact_response = re.sub(r"\s+", "", response)
        if self._contains_unsafe_output(response, compact_response):
            state["response"] = self.SAFE_REVIEW_FALLBACK
            state["safety_blocked"] = True
            state["safety_block_reason"] = "unsafe-medical-instruction"
            return state
        if "不代表诊断" not in response and state.get("risk_level") != "high":
            response += "\n\n提示：MedGuide 不替代医生诊断或处方。"
        state["response"] = response
        state["safety_blocked"] = False
        return state

    @classmethod
    def _contains_unsafe_output(cls, response: str, compact_response: str) -> bool:
        """Fail closed on actionable medical instructions, allowing refusals."""
        for candidate in (response, compact_response):
            safe_spans = [
                match.span()
                for pattern in cls._SAFE_NEGATED_OUTPUT
                for match in pattern.finditer(candidate)
            ]
            for pattern in cls._UNSAFE_OUTPUT_PATTERNS:
                for match in pattern.finditer(candidate):
                    start, end = match.span()
                    if any(start >= safe_start and end <= safe_end for safe_start, safe_end in safe_spans):
                        continue
                    return True
        return False

    def finalize(self, state: WorkflowState) -> WorkflowState:
        citations = state.get("citations", [])
        structured_result = state.get("structured_result")
        structured_success = isinstance(structured_result, dict) and not structured_result.get("blocked", False)
        state["confidence"] = round(min(0.96, 0.52 + 0.10 * len(citations) + (0.12 if structured_success else 0)), 2)
        response_id = f"{state.get('session_id', 'session')}:turn:{int(state.get('turn_count', 0))}"
        response_ids = list(state.get("response_ids", []))
        if response_id not in response_ids:
            response_ids.append(response_id)
        state["response_id"] = response_id
        state["response_ids"] = response_ids[-100:]
        profile = state.get("profile", {})
        summary_parts: list[str] = []
        if profile.get("age"):
            summary_parts.append(f"{profile['age']}岁")
        if profile.get("sex"):
            summary_parts.append(str(profile["sex"]))
        if profile.get("symptoms"):
            summary_parts.append("主要症状：" + "、".join(profile["symptoms"][:4]))
        if profile.get("duration"):
            summary_parts.append("持续" + str(profile["duration"]))
        if profile.get("history"):
            summary_parts.append("既往史：" + str(profile["history"]))
        state["summary"] = "；".join(summary_parts) if summary_parts else "已建立会话，等待补充症状和基础信息。"
        # The runner writes its actual engine after the node loop completes.
        state.setdefault("workflow_engine", "deterministic")
        return state
