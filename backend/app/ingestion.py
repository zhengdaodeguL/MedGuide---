from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import parse_qsl, unquote, urlsplit

from .embeddings import EmbeddingProvider, embedding_provider_from_env
from .models import KnowledgeDocument


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, str]


class _HTMLTextExtractor(HTMLParser):
    """Strip markup while suppressing executable or non-visible elements."""

    _SUPPRESSED = frozenset({"script", "style", "iframe", "object", "template", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppressed: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.lower()
        if normalized in self._SUPPRESSED:
            self._suppressed.append(normalized)
        if not self._suppressed:
            self.parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag, attrs
        if not self._suppressed:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if self._suppressed:
            if normalized == self._suppressed[-1]:
                self._suppressed.pop()
            if not self._suppressed:
                self.parts.append(" ")
            return
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._suppressed:
            self.parts.append(data)


class MedicalDocumentCleaner:
    """Normalize licensed source text before embedding; never invent missing clinical facts."""

    _whitespace = re.compile(r"\s+")
    # These patterns intentionally target explicit identity fields.  A broad
        # dictionary-based name scrubber would remove clinical terms (such as
    # drug and department names), so unknown free text is left intact for
    # human review instead of being silently corrupted.
    _private = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8}|\d{17}[\dXx])(?!\d)")
    _email = re.compile(r"(?<![\w.+-])[\w.+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w.-])")
    _named_field = re.compile(
        r"(?P<label>患者?(?:姓名|名字)|姓名|联系人(?:姓名)?|家属(?:姓名)?)"
        r"\s*(?:是|为|[:：#])?\s*(?P<value>[\u4e00-\u9fff]{2,6})"
    )
    _address_field = re.compile(
        r"(?P<label>家庭住址|居住地址|联系地址|住址|地址)\s*(?:是|为|[:：#])?\s*"
        r"(?P<value>[^，。；;\n]{4,80})"
    )
    _record_field = re.compile(
        r"(?P<label>(?:电子)?病历号|就诊号|住院号|门诊号|患者编号|病例号|病案号)"
        r"\s*(?:是|为|[:：#])?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9_\-/]{2,39})",
        re.IGNORECASE,
    )
    _contact_field = re.compile(
        r"(?P<label>微信号|QQ号|邮编|联系电话|联系方式)\s*(?:是|为|[:：#])?\s*"
        r"(?P<value>[A-Za-z0-9_+\-]{3,64})",
        re.IGNORECASE,
    )
    _inline_name = re.compile(
        r"(?P<label>患者|病人|联系人|家属)(?P<value>[\u4e00-\u9fff]{2,4})"
        r"(?=(?:现住|住在|居住|家住|[，,。；;\s]))"
    )
    _self_name = re.compile(
        r"(?P<label>(?:我|本人)\s*(?:叫|名叫)|我的名字\s*(?:是|为)?)\s*"
        r"(?P<value>[\u4e00-\u9fff]{2,6})(?=[，,。；;\s]|$)"
    )
    _self_name_before_clinical = re.compile(
        r"(?P<label>(?:我|本人)\s*(?:叫|名叫|是)|我的名字\s*(?:是|为)?)\s*"
        r"(?P<value>(?!(?:男性|女性|男士|女士|孕妇|患者|病人|孩子|宝宝|儿童|老人|老年人))"
        r"[\u4e00-\u9fff]{2,4}?)(?="
        r"今年|现年|患有|患上|得了|出现|主诉|发热|发烧|咳嗽|咽痛|腹痛|肚子痛|"
        r"头痛|头疼|胸痛|皮疹|头晕|恶心|呕吐|乏力|耳鸣|耳痛|流感|感冒|"
        r"鼻塞|流涕|腹泻|便秘|心悸|气促|呼吸困难|失眠|水肿|腰痛|关节痛|牙痛)"
    )
    _residence = re.compile(
        r"(?P<label>现住(?:在)?|住在|居住于?|家住)\s*"
        r"(?P<value>[^，。；;\n]{4,100})"
    )
    _free_address = re.compile(
        r"(?:北京市|上海市|天津市|重庆市|[\u4e00-\u9fff]{2,10}(?:省|自治区|市|区|县))"
        r"[\u4e00-\u9fffA-Za-z0-9-]{2,60}(?:路|街|道|巷|号|栋|单元|室)"
        r"[\u4e00-\u9fffA-Za-z0-9-]{0,20}"
    )
    _model_self_identity = re.compile(
        r"(?P<label>(?:我|本人)\s*是)\s*"
        r"(?P<value>(?!(?:男性|女性|男士|女士|孕妇|患者|病人|孩子|宝宝|儿童|老人|老年人|"
        r"医生|护士|学生)"
        r"(?=[，,。；;\s]|$))[\u4e00-\u9fff]{2,4})(?=[，,。；;\s]|$)"
    )
    _model_named_subject = re.compile(
        r"(?<![\u4e00-\u9fff])"
        r"(?P<value>(?!(?:患者|病人|本人|儿童|老人|老年人|孩子|宝宝|孕妇)\s*"
        r"(?:患有|患上|得了|出现|主诉))[\u4e00-\u9fff]{2,4})"
        r"(?=\s*(?:患有|患上|得了|出现|主诉))"
    )
    _metadata_inline_identity = re.compile(
        r"(?:患者|病人|联系人|家属)\s*[:：#]\s*[\u4e00-\u9fff]{2,6}(?=$|[，,。；;\s])"
    )
    _metadata_labeled_identity = re.compile(
        r"(?:姓名|名字|病历号|病例号|病案号|就诊号|住院号|门诊号|患者编号|"
        r"家庭住址|居住地址|联系地址|住址|地址|联系电话|联系方式|电子邮箱|邮箱|微信号|QQ号)"
        r"\s*(?:是|为|[:：#])\s*[^，。；;\s]{2,80}",
        re.IGNORECASE,
    )
    _EMBEDDING_QUERY_TERMS = frozenset(
        {
            "发热", "发烧", "低热", "高热", "高烧", "咳嗽", "咽痛", "喉咙痛", "腹痛", "肚子痛",
            "头痛", "头疼", "胸痛", "胸闷", "皮疹", "瘙痒", "头晕", "恶心", "呕吐", "乏力",
            "耳鸣", "耳痛", "鼻塞", "流涕", "腹泻", "便秘", "心悸", "气促", "气短", "呼吸困难",
            "呼吸不畅", "喘不上气", "喘不过气", "呼吸急促", "口唇发紫", "发绀", "失眠", "水肿",
            "腰痛", "关节痛", "牙痛", "昏厥", "晕厥", "意识模糊", "抽搐", "单侧无力", "偏瘫",
            "大量出血", "呕血", "黑便", "便血", "喉头紧", "脸唇肿", "感冒", "流感", "感染",
            "过敏", "过敏性休克", "症状", "体温", "疼痛", "男性", "女性", "男士", "女士", "孕妇",
            "怀孕", "妊娠", "孕期", "哺乳", "儿童", "婴幼儿", "老人", "老年人", "高血压", "糖尿病",
            "冠心病", "哮喘", "慢阻肺", "肾病", "肝病", "免疫抑制", "器官移植", "化疗", "既往史",
            "过敏史", "用药", "剂量", "副作用", "禁忌", "药品说明", "检查", "检验", "化验", "报告",
            "指标", "血常规", "胸片", "影像", "科室", "挂号", "排班", "急诊", "呼吸内科", "皮肤科",
            "全科医学科", "医保", "预约", "空腹", "注意事项", "库存", "价格", "费用",
        }
    )
    _EMBEDDING_QUERY_VALUES = (
        re.compile(r"(?<!\d)(?:[1-9]\d?|1[01]\d|120)\s*岁"),
        re.compile(r"(?<![\d一二两三四五六七八九十百两])(?:半|数|几|\d{1,2}|[一二两三四五六七八九十百两]{1,4})\s*(?:分钟|小时|天|周|个月|月|年)"),
        re.compile(r"(?<!\d)(?:3[5-9]|4[0-2])(?:\.\d)?\s*(?:摄氏度|℃|°C?|度)", re.IGNORECASE),
        re.compile(r"(?<!\d)\d{1,4}(?:\.\d+)?\s*(?:mmHg|mmol/L|mg/L|%)(?![\w/])", re.IGNORECASE),
    )
    _SENSITIVE_QUERY_KEYS = frozenset(
        {
            "accesskey",
            "accesstoken",
            "apikey",
            "auth",
            "authorization",
            "credential",
            "credentials",
            "jwt",
            "key",
            "password",
            "passwd",
            "privatekey",
            "secret",
            "session",
            "sessionid",
            "sig",
            "signature",
            "subscriptionkey",
            "token",
        }
    )
    _ALLOWED_SOURCE_HOSTS = frozenset(
        {
            "www.nhs.uk",
            "medlineplus.gov",
            "www.fda.gov",
            "telehealth.hhs.gov",
            "www.londonambulance.nhs.uk",
        }
    )
    _OPAQUE_ID = re.compile(r"[A-Za-z0-9._:#-]{1,256}")

    @classmethod
    def sanitize_opaque_id(cls, value: str, field: str = "id") -> str:
        """Bound identifiers to an ASCII syntax that excludes emails and URLs."""
        if not isinstance(value, str) or cls._OPAQUE_ID.fullmatch(value) is None:
            raise ValueError(
                f"{field} must be a 1-256 character ASCII opaque ID using only letters, digits, ._:#-"
            )
        return value

    @classmethod
    def _sensitive_query_key(cls, value: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", value).lower())
        return normalized in cls._SENSITIVE_QUERY_KEYS or any(
            marker in normalized
            for marker in (
                "accesskey",
                "apikey",
                "privatekey",
                "subscriptionkey",
                "token",
                "secret",
                "password",
                "credential",
                "signature",
            )
        )

    @classmethod
    def _contains_nested_credentials(cls, value: str) -> bool:
        decoded = value
        for _ in range(4):
            next_value = unquote(decoded)
            if next_value == decoded:
                break
            decoded = next_value
        normalized = unicodedata.normalize("NFKC", decoded)
        for match in re.finditer(r"(?:^|[?&;\s])([^=&?#\s]+)\s*=", normalized):
            if cls._sensitive_query_key(match.group(1)):
                return True
        return False

    @staticmethod
    def _strip_html(text: str) -> str:
        parser = _HTMLTextExtractor()
        parser.feed(text)
        parser.close()
        return "".join(parser.parts)

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self._strip_html(text)
        text = self._private.sub("[已脱敏]", text)
        text = self._email.sub("[邮箱已脱敏]", text)
        # Keep the field label so the resulting chunk remains understandable,
        # while replacing the identifying value before indexing or prompting.
        text = self._named_field.sub(lambda match: f"{match.group('label')}[已脱敏]", text)
        text = self._address_field.sub(lambda match: f"{match.group('label')}[已脱敏]", text)
        text = self._record_field.sub(lambda match: f"{match.group('label')}[已脱敏]", text)
        text = self._contact_field.sub(lambda match: f"{match.group('label')}[已脱敏]", text)
        text = self._inline_name.sub(lambda match: f"{match.group('label')}[已脱敏]", text)
        text = self._self_name_before_clinical.sub(
            lambda match: f"{match.group('label')}[已脱敏]",
            text,
        )
        text = self._self_name.sub(lambda match: f"{match.group('label')}[已脱敏]", text)
        text = self._residence.sub(lambda match: f"{match.group('label')}[已脱敏]", text)
        text = self._free_address.sub("[地址已脱敏]", text)
        text = self._whitespace.sub(" ", text)
        return text.strip()

    def sanitize_for_model(self, text: str) -> str:
        """Apply stricter identity removal at the external model boundary."""
        text = self.clean(text)
        text = self._model_self_identity.sub(lambda match: f"{match.group('label')}[已脱敏]", text)
        text = self._model_named_subject.sub("[姓名已脱敏]", text)
        return self._whitespace.sub(" ", text).strip()

    def sanitize_query_for_embedding(
        self,
        text: str,
        *,
        intent: str = "unknown",
        allowed_terms: Iterable[str] = (),
    ) -> str:
        """Build a PHI-minimized embedding query from allowlisted clinical fragments."""
        if not isinstance(text, str):
            text = ""
        normalized = unicodedata.normalize("NFKC", self._strip_html(text))
        terms = set(self._EMBEDDING_QUERY_TERMS)
        for value in allowed_terms:
            term = unicodedata.normalize("NFKC", str(value)).strip()
            if 2 <= len(term) <= 40 and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9+._/-]+", term):
                terms.add(term)

        candidates: list[tuple[int, int, str]] = []
        for term in sorted(terms, key=len, reverse=True):
            for match in re.finditer(re.escape(term), normalized, re.IGNORECASE):
                candidates.append((match.start(), match.end(), match.group(0)))
        for pattern in self._EMBEDDING_QUERY_VALUES:
            candidates.extend((match.start(), match.end(), match.group(0)) for match in pattern.finditer(normalized))

        selected: list[tuple[int, int, str]] = []
        for candidate in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
            start, end, _value = candidate
            if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end, _ in selected):
                continue
            selected.append(candidate)
        selected.sort(key=lambda item: item[0])

        fragments: list[str] = []
        for _start, _end, value in selected:
            compact = re.sub(r"\s+", "", value)
            if compact and compact not in fragments:
                fragments.append(compact)
            if len(fragments) >= 16:
                break

        intent_label = {
            "disease": "疾病症状",
            "drug": "药品信息",
            "exam": "检查检验",
            "department": "科室导诊",
            "faq": "就诊常见问题",
            "structured": "结构化查询",
        }.get(str(intent), "健康咨询")
        clinical = "".join(fragments)[:160] or "无可外发临床实体"
        return f"已脱敏查询；咨询类型：{intent_label}；临床要素：{clinical}"

    def sanitize_metadata(self, value: str, field: str) -> str:
        """Reject high-confidence identity fields before indexing or citation display."""
        if not isinstance(value, str):
            raise ValueError(f"knowledge metadata field {field} must be text")
        cleaned = self._whitespace.sub(" ", self._strip_html(value)).strip()
        normalized = unicodedata.normalize("NFKC", cleaned)
        if (
            self._private.search(normalized)
            or self._email.search(normalized)
            or self._metadata_inline_identity.search(normalized)
            or self._metadata_labeled_identity.search(normalized)
            or self._record_field.search(normalized)
            or self._contact_field.search(normalized)
            or self._free_address.search(normalized)
        ):
            raise ValueError(f"knowledge metadata field {field} contains sensitive identity data")
        # Body-text name heuristics can mistake clinical titles for identities.
        # Preserve vetted citation labels after explicit-field rejection above.
        if not cleaned:
            raise ValueError(f"knowledge metadata field {field} is empty after cleaning")
        return cleaned

    @staticmethod
    def sanitize_source_url(value: str) -> str:
        """Accept only public HTTPS citation URLs without embedded credentials."""
        if not isinstance(value, str):
            raise ValueError("knowledge metadata field source_url must be text")
        cleaned = value.strip()
        if not cleaned:
            return ""
        parsed = urlsplit(cleaned)
        try:
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ValueError("knowledge metadata field source_url must be a public HTTPS URL") from exc
        if (
            parsed.scheme.lower() != "https"
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or port not in (None, 443)
            or parsed.fragment
            or any(character.isspace() or ord(character) < 32 for character in cleaned)
        ):
            raise ValueError("knowledge metadata field source_url must be a public HTTPS URL")
        hostname = hostname.rstrip(".").lower()
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            labels = hostname.split(".")
            valid_hostname = (
                len(labels) >= 2
                and all(
                    label
                    and len(label) <= 63
                    and not label.startswith("-")
                    and not label.endswith("-")
                    and re.fullmatch(r"[a-z0-9-]+", label)
                    for label in labels
                )
            )
            if not valid_hostname or hostname.endswith(".localhost"):
                raise ValueError("knowledge metadata field source_url must be a public HTTPS URL")
        else:
            if not address.is_global:
                raise ValueError("knowledge metadata field source_url must be a public HTTPS URL")
        if hostname == "localhost":
            raise ValueError("knowledge metadata field source_url must be a public HTTPS URL")
        if hostname not in MedicalDocumentCleaner._ALLOWED_SOURCE_HOSTS:
            raise ValueError("knowledge metadata field source_url is not an allowed source")
        if re.search(r"%(?![0-9a-fA-F]{2})", parsed.query):
            raise ValueError("knowledge metadata field source_url contains an invalid query")
        for raw_key, raw_value in parse_qsl(parsed.query, keep_blank_values=True):
            key = raw_key
            value = raw_value
            for _ in range(3):
                decoded_key = unquote(key)
                decoded_value = unquote(value)
                if decoded_key == key and decoded_value == value:
                    break
                key, value = decoded_key, decoded_value
            if (
                MedicalDocumentCleaner._sensitive_query_key(key)
                or MedicalDocumentCleaner._contains_nested_credentials(value)
            ):
                raise ValueError("knowledge metadata field source_url contains a sensitive query parameter")
        return cleaned

    def sanitize_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Revalidate stored knowledge before it can become a citation."""
        if not isinstance(chunk, DocumentChunk):
            raise ValueError("knowledge index contains an invalid chunk")
        chunk_id = self.sanitize_opaque_id(chunk.id, "knowledge chunk id")
        document_id = self.sanitize_opaque_id(chunk.document_id, "knowledge document id")
        text = self.clean(chunk.text)
        if not text:
            raise ValueError("knowledge index contains an empty chunk")
        metadata = chunk.metadata
        if not isinstance(metadata, dict):
            raise ValueError("knowledge index contains invalid metadata")
        required = ("title", "category", "source", "updated_at")
        sanitized = {
            field: self.sanitize_metadata(metadata.get(field, ""), field)
            for field in required
        }
        source_url = self.sanitize_source_url(metadata.get("source_url", ""))
        if source_url:
            sanitized["source_url"] = source_url
        return DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            metadata=sanitized,
        )

    def chunks(self, document: KnowledgeDocument, chunk_size: int = 180, overlap: int = 30) -> list[DocumentChunk]:
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        text = self.clean(document.text)
        if not text:
            return []
        metadata = {
            "title": self.sanitize_metadata(document.title, "title"),
            "category": self.sanitize_metadata(document.category, "category"),
            "source": self.sanitize_metadata(document.source, "source"),
            "updated_at": self.sanitize_metadata(document.updated_at, "updated_at"),
        }
        if document.source_url:
            metadata["source_url"] = self.sanitize_source_url(document.source_url)
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                boundary = max(text.rfind("。", start, end), text.rfind("；", start, end), text.rfind(" ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    id=f"{document.id}#chunk-{index:03d}",
                    document_id=document.id,
                    text=chunk_text,
                    metadata=metadata,
                ))
                index += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def ingest(self, documents: Iterable[KnowledgeDocument], chunk_size: int = 180) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for document in documents:
            chunks.extend(self.chunks(document, chunk_size=chunk_size))
        return chunks


def load_catalog(path: Path) -> list[KnowledgeDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("knowledge catalog must contain a JSON list")
    documents: list[KnowledgeDocument] = []
    seen_ids: set[str] = set()
    cleaner = MedicalDocumentCleaner()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("knowledge catalog contains an invalid document")
        required = ("id", "title", "category", "source", "updated_at", "text")
        if any(not str(item.get(field, "")).strip() for field in required):
            raise ValueError("knowledge catalog contains an incomplete document")
        document_id = cleaner.sanitize_opaque_id(item["id"], "knowledge document id")
        if document_id in seen_ids:
            raise ValueError(f"knowledge catalog contains duplicate id: {document_id}")
        seen_ids.add(document_id)
        documents.append(
            KnowledgeDocument(
                id=document_id,
                title=str(item["title"]),
                category=str(item["category"]),
                source=str(item["source"]),
                updated_at=str(item["updated_at"]),
                text=str(item["text"]),
                source_url=MedicalDocumentCleaner.sanitize_source_url(str(item.get("source_url", ""))),
                tags=tuple(str(tag) for tag in item.get("tags", [])),
            )
        )
    return documents


def _stable_int64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def _ingestion_lock(index_path: Path) -> Iterator[None]:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_name(f".{index_path.name}.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another ingestion is active for {index_path}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _milvus_row(
    chunk: DocumentChunk,
    vector: list[float],
    adapter: object,
    corpus_generation: str,
) -> dict[str, object]:
    output_fields = set(getattr(adapter, "output_fields", ()))
    row: dict[str, object] = {str(getattr(adapter, "vector_field", "embedding")): vector}
    primary_field = str(getattr(adapter, "primary_field", ""))
    if "id" in output_fields and primary_field in {"", "id"}:
        row["id"] = _stable_int64(f"{corpus_generation}:{chunk.id}")
    if "chunk_id" in output_fields:
        row["chunk_id"] = chunk.id
    values = {
        "document_id": chunk.document_id,
        "text": chunk.text,
        "title": chunk.metadata.get("title", ""),
        "category": chunk.metadata.get("category", "unknown"),
        "source": chunk.metadata.get("source", ""),
        "updated_at": chunk.metadata.get("updated_at", ""),
        "source_url": chunk.metadata.get("source_url", ""),
        "corpus_generation": corpus_generation,
    }
    row.update({key: value for key, value in values.items() if key in output_fields})
    return row


def upsert_catalog(
    *,
    catalog_path: Path,
    bm25_index_path: Path,
    adapter: object,
    embedding_provider: EmbeddingProvider,
    chunk_size: int = 180,
    batch_size: int = 64,
) -> dict[str, object]:
    if chunk_size <= 30:
        raise ValueError("chunk_size must be greater than the 30 character overlap")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ensure_embedding_ready = getattr(embedding_provider, "ensure_ready", None)
    if callable(ensure_embedding_ready) and not ensure_embedding_ready():
        raise RuntimeError("embedding provider is unavailable")
    if not embedding_provider.available:
        raise RuntimeError("embedding provider is unavailable")
    ensure_ready = getattr(adapter, "ensure_ready", None)
    ready = bool(ensure_ready()) if callable(ensure_ready) else bool(getattr(adapter, "ready", False))
    if not ready:
        error = getattr(adapter, "error", None) or "MilvusUnavailable"
        raise RuntimeError(f"Milvus collection contract validation failed: {error}")

    chunks = MedicalDocumentCleaner().ingest(load_catalog(catalog_path), chunk_size=chunk_size)
    if not chunks:
        raise ValueError("knowledge catalog produced no chunks")
    upsert = getattr(adapter, "upsert", None)
    if not callable(upsert):
        raise RuntimeError("Milvus adapter does not support upsert")
    delete_stale = getattr(adapter, "delete_stale_generations", None)
    if not callable(delete_stale):
        raise RuntimeError("Milvus adapter does not support generation cleanup")
    from .bm25 import BM25Index

    bm25_index = BM25Index(chunks, embedding_provider.spec)
    corpus_generation = bm25_index.corpus_generation
    upserted = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedding_provider.embed([chunk.text for chunk in batch])
        if len(vectors) != len(batch) or any(
            len(vector) != embedding_provider.spec.dimension for vector in vectors
        ):
            raise RuntimeError("embedding provider returned an invalid batch")
        rows = [
            _milvus_row(chunk, vector, adapter, corpus_generation)
            for chunk, vector in zip(batch, vectors)
        ]
        count = int(upsert(rows))
        if count != len(rows):
            raise RuntimeError(f"Milvus acknowledged {count} of {len(rows)} chunks")
        upserted += count

    # Publish the lexical snapshot only after all vector batches succeed, so
    # readers never observe a newer BM25 generation than Milvus.
    bm25_index.save(bm25_index_path)
    deleted = int(delete_stale(corpus_generation))
    return {
        "documents": len({chunk.document_id for chunk in chunks}),
        "chunks": len(chunks),
        "upserted": upserted,
        "deleted_stale": deleted,
        "corpus_generation": corpus_generation,
        "bm25_index": str(bm25_index_path.resolve()),
        "embedding": embedding_provider.spec.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Clean, embed, and upsert a MedGuide knowledge catalog")
    parser.add_argument("--catalog", type=Path, default=root / "data" / "knowledge" / "catalog.json")
    parser.add_argument("--bm25-index", type=Path, default=root / "data" / "runtime" / "bm25-index.json")
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", "medguide_knowledge"))
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    uri = os.getenv("MILVUS_URI", "").strip()
    if not uri:
        parser.error("MILVUS_URI is required")
    try:
        provider = embedding_provider_from_env(production=True)
    except ValueError as exc:
        parser.error(str(exc))
    from .retrieval import MilvusAdapter

    adapter = MilvusAdapter(uri=uri, collection=args.collection, embedding_spec=provider.spec)
    try:
        with _ingestion_lock(args.bm25_index):
            result = upsert_catalog(
                catalog_path=args.catalog,
                bm25_index_path=args.bm25_index,
                adapter=adapter,
                embedding_provider=provider,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
            )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"ingestion failed: {exc}\n")
    finally:
        for resource in (adapter, provider):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
