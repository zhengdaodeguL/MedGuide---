from __future__ import annotations

import re

from .models import PatientProfile, RiskAssessment


class SafetyEngine:
    """Deterministic first-pass safety screen; LLM output never overrides this result."""

    RED_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "胸痛或胸部压榨感",
            (
                "胸痛", "胸口疼", "胸口痛", "胸闷", "胸闷痛", "胸部压榨", "胸口压迫",
                "胸部压迫感", "胸部有压迫感", "胸口发紧", "胸部发紧", "胸口憋闷",
                "胸部憋闷", "胸口闷", "胸口像被压住", "胸口被压住", "心前区痛", "心口疼",
            ),
        ),
        (
            "呼吸困难或口唇发紫",
            (
                "呼吸困难", "呼吸不畅", "无法呼吸", "不能呼吸", "呼吸不了", "上不来气",
                "气上不来", "喘不上气", "喘不上来气", "喘不过气", "喘不过来气", "吸不上气", "呼吸急促", "气促", "气短", "气喘",
                "喘憋", "口唇发紫", "嘴唇发紫", "口唇青紫", "发绀", "紫绀",
            ),
        ),
        ("意识改变或抽搐", ("昏厥", "晕厥", "晕倒", "昏倒", "意识模糊", "意识不清", "叫不醒", "无法唤醒", "无意识", "抽搐", "失去意识")),
        (
            "突发单侧无力或言语不清",
            (
                "口角歪", "脸歪嘴斜", "言语不清", "说话含糊", "说话不清楚",
                "说话突然不清楚", "突然说话不清楚", "一侧无力", "单侧无力", "单侧肢体无力",
                "一边身体没力气", "一边身子没力气", "半边身体没力气", "半身不遂", "偏瘫",
            ),
        ),
        (
            "大量出血或黑便",
            ("大量出血", "出血不止", "喷射性出血", "呕血", "吐血", "黑便", "黑色大便", "大便发黑", "便血不止"),
        ),
        ("严重过敏反应", ("喉头紧", "喉咙紧", "喉头发紧", "脸唇肿", "嘴唇肿", "面部肿胀", "呼吸道肿胀", "过敏性休克")),
        ("高热伴精神状态异常", ("高热不退", "高烧不退", "高热", "高烧", "精神状态异常", "高热伴意识")),
        ("突然剧烈头痛或腹痛", ("突然剧烈头痛", "突发剧烈头痛", "突然剧烈腹痛", "突发剧烈腹痛", "视力丧失")),
        ("异常呕吐物需紧急评估", ("咖啡渣样呕吐", "咖啡渣样的呕吐", "绿色呕吐物", "呕吐物是绿色", "呕吐物呈绿色")),
    )
    WATCH_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("症状持续或反复", ("反复", "持续", "越来越", "加重")),
        ("可能需要线下检查", ("发热", "腹痛", "头痛", "皮疹", "咳嗽")),
    )

    # These are phrase-level cues.  A bare ``不`` is intentionally handled by
    # the anchored regex in ``_asserted`` below; treating it as a substring
    # would mistake ``不舒服，胸痛`` for a negation of the later chest pain.
    _NEGATION = ("没有", "无", "否认", "不曾", "从未", "不是", "并非", "未见", "没", "不再")
    _QUESTION = (
        "是否",
        "有没有",
        "有无",
        "有哪些表现",
        "哪些表现",
        "什么表现",
        "什么是",
        "是什么",
        "什么意思",
        "吗",
        "？",
        "?",
    )
    _HELP_SEEKING = (
        "怎么办",
        "如何处理",
        "需要就医",
        "需要急诊",
        "要去医院",
        "该去医院",
        "该挂什么",
        "严重吗",
        "危险吗",
        "要紧吗",
    )
    _THIRD_PARTY = re.compile(
        r"朋友|家人|父亲|母亲|爸爸|妈妈|儿子|女儿|丈夫|妻子|老公|老婆|同事|邻居|"
        r"男朋友|女朋友|他人|他们|她们|(?:他|她)(?=(?:也|都|又|还|刚|今天|昨天|"
        r"最近|目前|现在|正在|已经|突然|有|出现|发生|感觉|感到|觉得|说|告诉|吃|服|"
        r"用|被|患|药|胸|呼吸|喘|气|吐|呕|黑|一侧|单侧|半身|口|脸|喉|高热|高烧|"
        r"昏|晕|抽|妈|爸|妻|夫))"
    )
    _PATIENT_MARKER = re.compile(r"我|本人|患者|自己|孩子|宝宝|孕妇")
    _HYPOTHETICAL = re.compile(
        r"(?:如果|假如|假设|万一|要是|若是|倘若|一旦)[^，,。；;！？!?]{0,30}$"
    )
    _KNOWLEDGE_PREFIX = re.compile(
        r"(?:只是|仅仅|单纯)?(?:想|要)?(?:问|了解|咨询|查询|查|科普)(?:一下|下|关于)?\s*$"
    )
    _KNOWLEDGE_SUFFIX = re.compile(
        r"^(?:的)?(?:表现|症状|定义|含义|意思|原因|常识|知识|鉴别)"
        r"(?:是什么|有哪些|有何|包括什么|怎么回事)?(?:吗|呢)?$"
    )
    _RESOLVED_BEFORE = re.compile(
        r"(?:已经|已|现已)(?:于[^，,。；;！？!?]{0,10})?(?:缓解|消失|好转|恢复)(?:的)?\s*$"
    )
    _RESOLVED_AFTER = re.compile(
        r"^(?:[，,。；;]\s*(?:但是|但)?\s*(?:我|本人|患者|症状)?\s*(?:目前|现在|如今)?)?"
        r"(?:已经|已|现已|目前已(?:经)?|现在已(?:经)?)(?:于[^，,。；;！？!?]{0,10})?"
        r"(?:明显|完全)?"
        r"(?:缓解|消失|好转|恢复)"
    )
    _CLAUSE_BREAK = re.compile(r"[，,。；;！？!?]|但是|但|可是|然而|却")
    _DIRECT_NEGATION = re.compile(
        r"(?:没有|无|否认|不曾|从未|不是|并非|未见|没|不再|未|不伴有|不伴随)"
        r"(?:(?:明显|严重|出现|发生|感到|觉得|有|伴有|伴随|存在|过))*$"
    )
    _BARE_NOT_NEGATION = re.compile(r"不(?:再|曾|会|伴有|伴随|觉得|感到|存在|出现|是|属于)?$")
    _TEMPERATURE_NEGATION = re.compile(
        r"(?:"
        r"(?:没有|没|未|从未|不曾)(?:达到|到|超过|高于|升到|升至|测到|测得)?"
        r"|低于|小于|少于|不到|不超过|不高于|未达到|没有达到|没达到|"
        r"没有到|没到|没超过|未超过|不满|不足"
        r")$"
    )
    _TEMPERATURE_QUESTION = re.compile(r"(?:是否|有无|有没有)[^，。；;！？!?]{0,8}$")

    @classmethod
    def _asserted(cls, text: str, pattern: str) -> bool:
        """Return whether a phrase describes the patient rather than a query/negation."""
        for match in re.finditer(re.escape(pattern), text):
            before = text[: match.start()]
            after = text[match.end():]
            # Restrict cues to the current clause.  In particular, the single
            # character ``不`` must be immediately attached to the symptom (or
            # a known negating verb), never merely present somewhere nearby.
            clause_before = cls._CLAUSE_BREAK.split(before)[-1]
            clause_after = cls._CLAUSE_BREAK.split(after)[0]
            full_clause = f"{clause_before}{pattern}{clause_after}"
            # A symptom attributed to a relative or posed only as a
            # hypothetical must not be recorded as the current patient's
            # emergency signal.  The last subject marker wins within a clause;
            # explicit "我/患者" after a relative re-attaches the statement.
            subject_markers = [
                (match, True)
                for match in cls._THIRD_PARTY.finditer(f"{clause_before}{pattern}")
                if match.start() < len(clause_before)
            ] + [
                (match, False) for match in cls._PATIENT_MARKER.finditer(clause_before)
            ]
            if subject_markers:
                last_subject, is_third_party = max(subject_markers, key=lambda item: item[0].start())
                if is_third_party:
                    patient_after_relative = cls._PATIENT_MARKER.search(clause_before[last_subject.end():])
                    if not patient_after_relative:
                        continue
            if cls._HYPOTHETICAL.search(clause_before):
                continue
            if cls._DIRECT_NEGATION.search(clause_before) or cls._BARE_NOT_NEGATION.search(clause_before):
                continue
            if cls._RESOLVED_BEFORE.search(clause_before):
                continue
            resolved = cls._RESOLVED_AFTER.match(after)
            if resolved and not re.search(
                r"(?:复发|加重|未完全(?:缓解|消失|好转|恢复)|"
                r"(?:又|再次|重新)[^，,。；;！？!?]{0,12}(?:出现|发作|疼|痛|闷|喘|呼吸|无力|出血|黑便))",
                after[resolved.end():],
            ):
                continue
            if re.match(r"(?:不再|已经没有|已没有|未见|不存在)", clause_after):
                continue
            # Coordinated phrases often put one negator after several
            # symptoms (``胸痛和呼吸困难都没有``) or before the list
            # (``没有胸痛和呼吸困难``).  Treat that negator as applying to
            # the complete list, while punctuation/conjunctions remain hard
            # scope boundaries for independent later symptoms.
            if re.search(
                r"(?:没有|无|否认|不曾|从未|不是|并非|未见|没|不再|未|不伴有|不伴随|不存在)"
                r"[^，。；;！？!?]{0,20}(?:和|或|及|与|、|以及)$",
                clause_before,
            ):
                continue
            if re.search(
                r"(?:和|或|及|与|、|以及|并且)[^，。；;！？!?]{0,30}"
                r"(?:都|均|全部)?(?:没有|无|否认|未(?:见|有|出现|发生)?|不存在|不伴有|不伴随)$",
                full_clause,
            ):
                continue
            if any(clause_before.endswith(marker) for marker in cls._QUESTION if marker not in ("吗", "？", "?")):
                continue
            knowledge_prefix = cls._KNOWLEDGE_PREFIX.search(clause_before)
            knowledge_suffix = cls._KNOWLEDGE_SUFFIX.fullmatch(clause_after)
            if (knowledge_prefix or (knowledge_suffix and not cls._PATIENT_MARKER.search(clause_before))) and not any(
                marker in full_clause for marker in cls._HELP_SEEKING
            ):
                continue
            is_question = any(marker in clause_after for marker in ("吗", "？", "?")) or any(
                clause_after.startswith(marker) for marker in cls._QUESTION if marker not in ("吗", "？", "?")
            )
            if is_question and not any(marker in full_clause for marker in cls._HELP_SEEKING):
                continue
            return True
        return False

    def screen(self, text: str, profile: PatientProfile | None = None) -> RiskAssessment:
        normalized = re.sub(r"\s+", "", text.lower())
        flags: list[str] = []
        for label, patterns in self.RED_FLAGS:
            if any(self._asserted(normalized, pattern) for pattern in patterns):
                flags.append(label)

        profile = profile or {}
        age = profile.get("age")
        try:
            age = int(age) if age is not None else None
        except (TypeError, ValueError):
            age = None
        if age is not None and (age < 3 or age >= 75):
            flags.append("婴幼儿或高龄人群")
        if any(self._asserted(normalized, word) for word in ("怀孕", "妊娠", "孕期", "孕妇", "哺乳")):
            flags.append("孕期或哺乳期特殊人群")
        if any(self._asserted(normalized, word) for word in ("化疗", "免疫抑制", "器官移植")):
            flags.append("免疫功能受影响人群")

        # Normalize common numeric temperature expressions so ``39.5℃`` and
        # ``40 度`` receive the same conservative high-risk treatment as
        # ``高热``.  The assertion helper still excludes negated/questioned
        # values (such as "没有 39 度").
        for match in re.finditer(
            r"(?<!\d)(3[89](?:\.\d+)?|4\d(?:\.\d+)?)\s*(?:摄氏度|℃|°c?|度)",
            normalized,
        ):
            value = float(match.group(1))
            before = normalized[max(0, match.start() - 16): match.start()]
            clause_before = self._CLAUSE_BREAK.split(before)[-1]
            after = normalized[match.end(): match.end() + 16]
            clause_after = self._CLAUSE_BREAK.split(after)[0]
            is_negated = bool(
                self._TEMPERATURE_NEGATION.search(clause_before)
                or re.match(r"(?:以下|以内|之下|不到)", clause_after)
            )
            is_question = bool(
                self._TEMPERATURE_QUESTION.search(clause_before)
                or clause_after.startswith(("吗", "是否", "是不是", "有无", "有没有"))
                or after.startswith(("？", "?"))
            ) and not any(marker in f"{clause_before}{match.group(0)}{clause_after}" for marker in self._HELP_SEEKING)
            if value >= 39 and not is_negated and not is_question:
                flags.append("高热伴精神状态异常")
                break

        if flags:
            advice = "建议立即联系当地急救服务或前往最近的急诊；不要仅依赖线上问答等待观察。"
            return RiskAssessment("high", tuple(dict.fromkeys(flags)), advice)

        watch_flags = [
            label
            for label, patterns in self.WATCH_FLAGS
            if any(self._asserted(normalized, pattern) for pattern in patterns)
        ]
        if watch_flags:
            return RiskAssessment("watch", tuple(dict.fromkeys(watch_flags)), "建议在 24 小时内关注变化，必要时预约线下就诊。")
        return RiskAssessment("low", (), "当前未识别到明确的高风险信号；如症状变化或加重，请及时就医。")
