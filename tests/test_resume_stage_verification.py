"""Intermediate GPT restart points must preserve training metadata."""

import pytest
import torch

from rebuild_gpt.resume_multistage import compare_multistage_resume
from rebuild_gpt.resume_stage_verification import verify_multistage_checkpoints


def _run(tmp_path):
    corpus = tmp_path / "data.txt"
    corpus.write_text("abba" * 55, encoding="utf-8")
    report = compare_multistage_resume(
        corpus, tmp_path / "runs", total_steps=4,
        stops=(1, 2), seed=5,
    )
    return corpus, report


def test_each_interruption_has_correct_step_and_matching_settings(tmp_path):
    corpus, run = _run(tmp_path)
    result = verify_multistage_checkpoints(run, corpus)
    assert [record["step"] for record in result["stages"]] == [1, 2, 4]
    assert all(len(record["sha256"]) == 64 for record in result["stages"])
    assert all(record["moment_parameters"] > 0 for record in result["stages"])
    assert result["final_comparison"]["identical"]


def test_corrupt_intermediate_corpus_identity_is_detected(tmp_path):
    corpus, run = _run(tmp_path)
    intermediate = run["stage_checkpoints"][1]
    payload = torch.load(intermediate, weights_only=True)
    payload["corpus_sha256"] = "0" * 64
    torch.save(payload, intermediate)
    with pytest.raises(ValueError, match="corpus identity"):
        verify_multistage_checkpoints(run, corpus)


def test_stage_list_missing_an_interruption_is_rejected(tmp_path):
    corpus, run = _run(tmp_path)
    run["stage_checkpoints"].pop(0)
    with pytest.raises(ValueError, match="count"):
        verify_multistage_checkpoints(run, corpus)
