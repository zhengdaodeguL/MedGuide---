from __future__ import annotations

import pytest

from app.retrieval import HybridRetriever
from app.safety import SafetyEngine
from app.sql_guard import ReadOnlySQLGuard
from app.workflow import MedGuideWorkflow


def make_workflow() -> MedGuideWorkflow:
    return MedGuideWorkflow(HybridRetriever(), ReadOnlySQLGuard(), SafetyEngine())


@pytest.mark.parametrize(
    "response",
    (
        "最好停药",
        "最好停止服用布洛芬",
        "推荐停药",
        "推荐服用布洛芬",
        "可停药",
        "可服用布洛芬",
        "应停药",
        "需要停药",
        "每次服2片",
        "每次服用2片",
        "服用2片",
        "停止布洛芬",
        "停布洛芬",
        "我不能诊断，但推荐服用布洛芬",
        "最好服用 Amoxicillin",
        "每次服 2 片",
        "服 用 2 片",
        "请停止服用 Ibuprofen",
    ),
)
def test_blocks_actionable_medication_language(response: str) -> None:
    result = make_workflow().safety_review({"risk_level": "low", "response": response})

    assert result["safety_blocked"] is True, response
    assert result["safety_block_reason"] == "unsafe-medical-instruction"


@pytest.mark.parametrize(
    "response",
    (
        "布洛芬属于非甾体抗炎药，常见不良反应包括胃部不适。",
        "服用布洛芬可能出现胃部不适。",
        "服用布洛芬的禁忌包括活动性消化道溃疡。",
        "服用布洛芬有哪些禁忌？",
        "布洛芬与其他药物有哪些相互作用？",
        "不能自行停药，请咨询医生或药师。",
        "不建议自行服用布洛芬。",
    ),
)
def test_keeps_information_questions_and_refusals(response: str) -> None:
    result = make_workflow().safety_review({"risk_level": "low", "response": response})

    assert result["safety_blocked"] is False, response


@pytest.mark.parametrize(
    "text",
    (
        "我胸口像被压住一样",
        "上不来气",
        "气上不来",
        "喘不上来气",
        "喘不过来气",
        "胸口闷",
        "胸部发紧",
        "胸部有压迫感",
        "我吐血了",
        "我黑色大便",
        "我一边身体没力气",
        "我说话突然不清楚",
        "我喉咙肿无法呼吸",
        "我无法呼吸",
        "我呼吸不了",
        "我吃他汀后胸痛",
        "我服用他莫昔芬后胸痛",
        "患者用药他汀，胸闷",
    ),
)
def test_current_patient_red_flag_synonyms_are_high(text: str) -> None:
    assert SafetyEngine().screen(text).level == "high", text


@pytest.mark.parametrize(
    "text",
    (
        "我家人胸痛，我没有不舒服",
        "家人突然胸痛怎么办",
        "他胸痛怎么办",
        "他今天胸痛怎么办",
        "她最近胸闷怎么办",
        "如果出现胸痛怎么办",
        "假如我现在胸闷该怎么办",
        "我只是问胸痛的表现",
        "想了解胸痛的表现是什么",
        "胸痛已经缓解",
        "我昨天胸痛，现在已经缓解",
        "我昨天胸痛，我现在已经明显缓解",
        "胸痛已经缓解，但仍需复查",
        "我吃他汀，没有胸痛",
        "我他汀过敏了",
    ),
)
def test_non_current_context_does_not_raise_patient_risk(text: str) -> None:
    assert SafetyEngine().screen(text).level != "high", text


@pytest.mark.parametrize(
    "text",
    (
        "我现在胸痛怎么办",
        "家人胸痛，但我现在也胸闷",
        "如果家人胸痛，我现在无法呼吸",
        "我只是想了解胸痛，但我现在喘不过来气",
        "胸痛已经缓解，但现在再次胸痛",
        "胸痛已经缓解，但现在又加重",
    ),
)
def test_explicit_current_patient_symptoms_still_raise_risk(text: str) -> None:
    assert SafetyEngine().screen(text).level == "high", text
