"""Registry (configs/data_sources.yaml) governance and facts tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "data_sources.yaml"

CODESWITCH = "Seif-Eldeen-Sameh/asr_codeswitched_dataset"
EGYSPEAK = "MohamedGomaa30/EGYSpeak"
CASABLANCA = "UBC-NLP/Casablanca"


@pytest.fixture(scope="module")
def registry() -> dict:
    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def datasets(registry: dict) -> dict[str, dict]:
    return {ds["id"]: ds for ds in registry["datasets"]}


class TestRegistry:
    def test_retrieval_date(self, registry: dict) -> None:
        assert registry["retrieved"] == "2026-09-20"

    def test_exact_revisions(self, datasets: dict) -> None:
        assert datasets[CODESWITCH]["revision"] == ("712de01079517771f95bcdecee68ca232b979628")
        assert datasets[EGYSPEAK]["revision"] == ("59c40fc6833382a743c180f50addcda5edaa47c3")
        assert datasets[CASABLANCA]["revision"] == ("8951b1b88e28c1107142ced57967b8d16350951d")

    def test_row_counts_and_roles(self, datasets: dict) -> None:
        assert datasets[CODESWITCH]["row_count"] == 45189
        assert datasets[EGYSPEAK]["row_count"] == 147979
        assert datasets[CODESWITCH]["role"] == "primary_training"
        assert datasets[EGYSPEAK]["role"] == "optional_pseudo_labelled_supplement"
        assert datasets[CASABLANCA]["role"] == "external_evaluation_only"

    def test_codeswitch_license_governance(self, datasets: dict) -> None:
        license = datasets[CODESWITCH]["license"]
        assert "not uniformly MIT" in license["caveat"].replace("NOT uniformly", "not uniformly")
        assert "12,480" in license["caveat"] or "12480" in license["caveat"]
        policy = datasets[CODESWITCH]["redistribution_policy"]
        assert policy.startswith("none")
        assert "do not redistribute" in policy.lower()
        blocker = datasets[CODESWITCH]["release_blocker"]
        assert blocker["blocked"] is True
        assert "source-chain review" in blocker["reason"] or "source review" in blocker["reason"]

    def test_egyspeak_pseudo_labelled_and_blocked(self, datasets: dict) -> None:
        ds = datasets[EGYSPEAK]
        assert ds["pseudo_labelled"] is True
        assert "EgypTalk-ASR-v2" in ds["card_metadata"]["transcript_generator"]
        license = ds["license"]
        assert "CC-BY-4.0" in license["declared"]
        assert "GPL-3.0" in license["caveat"]
        assert ds["release_blocker"]["blocked"] is True
        assert (
            "fail-closed" in ds["release_blocker"]["reason"]
            or "conflict" in ds["release_blocker"]["reason"]
        )

    def test_casablanca_eval_only(self, datasets: dict) -> None:
        ds = datasets[CASABLANCA]
        assert ds["splits"] == {"validation": 846, "test": 846}
        assert ds["config"] == "Egypt"
        assert ds["license"]["declared"] == "CC-BY-NC-ND-4.0"
        assert ds["pseudo_labelled"] is False
        assert (
            "never train" in " ".join(ds["allowed_use"]) or "evaluation_only" in ds["allowed_use"]
        )
        assert "aggregate metrics" in ds["redistribution_policy"]
        assert ds["release_blocker"]["blocked"] is True

    def test_no_invented_measured_facts(self, registry: dict) -> None:
        raw = REGISTRY_PATH.read_text(encoding="utf-8")
        # The registry holds card-declared facts only; no measured audit values.
        assert "total_seconds" not in raw
        assert "NOT RUN" not in raw or "measured" not in raw.lower().split("NOT RUN")[0][-30:]

    def test_no_pii_or_audio_in_registry(self, registry: dict) -> None:
        raw = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "wav" not in raw.lower() or "tar shards" in raw
        assert len(registry["datasets"]) == 3
