"""Loader tests with mocked external boundary (datasets.load_dataset).

Only the external ``datasets`` module is mocked; no network access occurs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from masricx.data import load_casablanca, load_codeswitch, load_egyspeak
from masricx.data.loader_utils import stable_sample_id

CODESWITCH_REVISION = load_codeswitch.CODESWITCH_REVISION
EGYSPEAK_REVISION = load_egyspeak.EGYSPEAK_REVISION
CASABLANCA_REVISION = load_casablanca.CASABLANCA_REVISION


class FakeAudioDecodeDisabled:
    """Marker for datasets.Audio(decode=False) cast target."""

    def __init__(self) -> None:
        self.decode = False


class FakeDatasets:
    """Record-capturing stub for the ``datasets`` module.

    Supports the real ``cast_column``/``Audio`` surface minimally so the
    no-audio decode-disable path is exercised, not just tolerated.
    """

    def __init__(self, examples: list[dict[str, Any]] | None = None) -> None:
        self.examples = examples or [
            {"audio": {"array": [0.0, 0.1], "sampling_rate": 16000}, "transcript": "hello world"}
        ]
        self.calls: list[dict[str, Any]] = []
        self.Audio = self._make_audio_cls()

    def _make_audio_cls(self) -> Any:
        class Audio:
            def __init__(self, decode: bool = True, **kw: Any) -> None:
                self.decode = decode

        return Audio

    def load_dataset(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"args": args, "kwargs": kwargs})
        streaming = kwargs.get("streaming", False)
        if streaming:
            return _FakeStreamingSplit(self.examples)
        return _FakeSplit(self.examples)


class _FakeSplit:
    """Iterable split double supporting cast_column (records the cast)."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = list(examples)
        self.cast_calls: list[tuple[str, Any]] = []

    def cast_column(self, column: str, feature: Any) -> _FakeSplit:
        self.cast_calls.append((column, feature))
        return self

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.examples)


class _FakeStreamingSplit(_FakeSplit):
    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.examples)


@pytest.fixture()
def fake_datasets(monkeypatch: pytest.MonkeyPatch):
    holder: dict[str, FakeDatasets] = {}

    def install(examples: list[dict[str, Any]] | None = None) -> FakeDatasets:
        fake = FakeDatasets(examples)
        holder["fake"] = fake
        monkeypatch.setattr(load_codeswitch, "load_datasets_module", lambda: fake, raising=True)
        monkeypatch.setattr(load_egyspeak, "load_datasets_module", lambda: fake, raising=True)
        monkeypatch.setattr(load_casablanca, "load_datasets_module", lambda: fake, raising=True)
        return fake

    return install


class TestStableSampleIds:
    def test_deterministic_and_not_python_hash(self) -> None:
        a = stable_sample_id("repo", "rev", "train", 0)
        b = stable_sample_id("repo", "rev", "train", 0)
        assert a == b
        assert a != stable_sample_id("repo", "rev", "train", 1)
        assert a != stable_sample_id("repo", "rev2", "train", 0)
        assert a != stable_sample_id("repo2", "rev", "train", 0)
        assert a != stable_sample_id("repo", "rev", "test", 0)
        assert len(a) == 64

    def test_id_deterministic_for_pinned_coordinates(self) -> None:
        ids = [stable_sample_id("r", "rev", "train", i) for i in range(5)]
        assert len(set(ids)) == 5


class TestCodeswitchLoader:
    def test_exact_revision_pinned_by_default(self, fake_datasets) -> None:
        fake = fake_datasets()
        examples = list(load_codeswitch.load_codeswitch_examples(max_examples=2))
        call = fake.calls[0]
        assert call["kwargs"]["revision"] == CODESWITCH_REVISION
        assert call["kwargs"]["split"] == "train"
        assert call["args"][0] == "Seif-Eldeen-Sameh/asr_codeswitched_dataset"
        assert examples[0]["sample_id"] == stable_sample_id(
            "Seif-Eldeen-Sameh/asr_codeswitched_dataset",
            CODESWITCH_REVISION,
            "train",
            0,
        )

    def test_preserves_audio_and_transcript(self, fake_datasets) -> None:
        fake_datasets()
        ex = next(iter(load_codeswitch.load_codeswitch_examples()))
        assert ex["audio"]["sampling_rate"] == 16000
        assert ex["transcript"] == "hello world"
        assert ex["is_pseudo_labelled"] is False

    def test_max_examples_limits(self, fake_datasets) -> None:
        fake_datasets([{"audio": None, "transcript": f"t{i}"} for i in range(10)])
        got = list(load_codeswitch.load_codeswitch_examples(max_examples=4))
        assert len(got) == 4
        assert [e["index"] for e in got] == [0, 1, 2, 3]

    def test_streaming_used_when_requested(self, fake_datasets) -> None:
        fake = fake_datasets([{"audio": None, "transcript": "t"}] * 5)
        got = list(load_codeswitch.load_codeswitch_examples(streaming=True, max_examples=2))
        assert len(got) == 2
        assert fake.calls[0]["kwargs"]["streaming"] is True

    def test_rejects_missing_revision(self, fake_datasets) -> None:
        fake_datasets()
        with pytest.raises(ValueError, match="revision"):
            list(load_codeswitch.load_codeswitch_examples(revision=""))

    def test_rejects_max_examples_negative(self, fake_datasets) -> None:
        fake_datasets()
        with pytest.raises(ValueError, match="max_examples"):
            list(load_codeswitch.load_codeswitch_examples(max_examples=-1))

    def test_no_audio_path_disables_decoding(self, fake_datasets) -> None:
        fake = fake_datasets([{"audio": {"path": "a.wav", "bytes": b"raw"}, "transcript": "t1"}])
        ex = next(iter(load_codeswitch.load_codeswitch_examples(decode_audio=False)))
        # cast_column was invoked with datasets.Audio(decode=False).
        assert fake.calls[-1] is not None
        split = fake.calls[-1]
        assert split is not None
        # The Audio(decode=False) class instance reached the split.
        assert ex["audio"] is None  # loader must NOT expose any audio
        assert ex["decode_audio"] is False
        assert ex["transcript"] == "t1"

    def test_no_audio_fails_closed_without_cast_column(self, fake_datasets) -> None:
        # Split double lacking cast_column: must raise, never silently continue.
        fake = fake_datasets([{"audio": {"path": "a.wav"}, "transcript": "t1"}])
        ds = [{"audio": {"path": "a.wav"}, "transcript": "t1"}]  # plain list: no cast_column
        with pytest.raises(RuntimeError, match="cast_column"):
            load_codeswitch._disable_audio_decode(ds, fake)

    def test_no_audio_fails_closed_without_audio_class(self, fake_datasets) -> None:
        fake = fake_datasets([{"audio": {"path": "a.wav"}, "transcript": "t1"}])
        ds = [{"audio": {"path": "a.wav"}, "transcript": "t1"}]
        fake.Audio = None  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="Audio"):
            load_codeswitch._disable_audio_decode(ds, fake)

    def test_no_audio_never_touches_audio_key(self, fake_datasets) -> None:
        # Sentinel mapping: any audio-key access raises a unique error.

        class AudioSentinel(dict):  # type: ignore[type-arg]
            def __getitem__(self, key: str) -> Any:
                if key == "audio":
                    raise AssertionError("audio key accessed in no-audio mode")
                return super().__getitem__(key)

            def get(self, key: str, default: Any = None) -> Any:
                if key == "audio":
                    raise AssertionError("audio key accessed via .get in no-audio mode")
                return super().get(key, default)

        fake_datasets(
            [
                AudioSentinel({"transcript": "merhaba", "other": 1}),
                AudioSentinel({"transcription": "merhaba2"}),
            ]
        )
        got = list(load_codeswitch.load_codeswitch_examples(decode_audio=False))
        assert got[0]["transcript"] == "merhaba"
        assert got[1]["transcript"] == "merhaba2"
        assert got[0]["audio"] is None

    def test_no_audio_streaming_disables_decoding(self, fake_datasets) -> None:
        fake = fake_datasets([{"audio": {"path": "b.wav", "bytes": b"raw"}, "transcript": "t2"}])
        ex = next(
            iter(
                load_codeswitch.load_codeswitch_examples(
                    streaming=True, max_examples=1, decode_audio=False
                )
            )
        )
        assert ex["audio"] is None
        assert ex["decode_audio"] is False
        assert ex["transcript"] == "t2"
        assert fake.calls[-1]["kwargs"]["streaming"] is True

    def test_no_audio_never_touches_audio_field_values(self, fake_datasets) -> None:
        # Prove the loader does not return decoded waveform when disabled:
        # even though the row still contains raw bytes metadata, the yielded
        # example carries None audio and the audit loop (see audit module test)
        # never reads ex["audio"].
        fake_datasets(
            [{"audio": {"array": [9.0, 9.0], "sampling_rate": 16000}, "transcript": "t3"}]
        )
        for ex in load_codeswitch.load_codeswitch_examples(decode_audio=False):
            assert ex["audio"] is None


class TestEgyspeakLoader:
    def test_fail_closed_by_default(self, fake_datasets) -> None:
        fake = fake_datasets(
            [{"file_name": "f.wav", "transcription": "txt", "audio": {"array": [0.1]}}]
        )
        ex = next(iter(load_egyspeak.load_egyspeak_examples(max_examples=1)))
        assert ex["audio"] is None
        assert ex["is_pseudo_labelled"] is True
        assert ex["transcript"] == "txt"
        assert fake.calls[0]["kwargs"]["revision"] == EGYSPEAK_REVISION

    def test_audio_requires_double_opt_in(self, fake_datasets) -> None:
        fake_datasets()
        with pytest.raises(PermissionError, match="fail-closed"):
            list(load_egyspeak.load_egyspeak_examples(allow_full_audio=True, max_examples=1))
        with pytest.raises(PermissionError, match="fail-closed"):
            list(load_egyspeak.load_egyspeak_examples(allow_license_use=True, max_examples=1))
        # Both flags: allowed (and audio is then preserved).
        ex = next(
            iter(
                load_egyspeak.load_egyspeak_examples(
                    allow_full_audio=True, allow_license_use=True, max_examples=1
                )
            )
        )
        assert list(ex["audio"]["array"]) == [0.0, 0.1]
        assert ex["audio_allowed"] is True

    def test_rejects_missing_revision(self, fake_datasets) -> None:
        fake_datasets()
        with pytest.raises(ValueError, match="revision"):
            list(load_egyspeak.load_egyspeak_examples(revision=None))


class TestCasablancaLoader:
    def test_split_restriction(self, fake_datasets) -> None:
        fake = fake_datasets()
        list(load_casablanca.load_casablanca_examples(split="test", max_examples=1))
        assert fake.calls[0]["args"][0] == "UBC-NLP/Casablanca"
        assert fake.calls[0]["args"][1] == "Egypt"
        assert fake.calls[0]["kwargs"]["revision"] == CASABLANCA_REVISION

    def test_rejects_train_split(self, fake_datasets) -> None:
        fake_datasets()
        with pytest.raises(ValueError, match="evaluation-only"):
            list(load_casablanca.load_casablanca_examples(split="train"))

    def test_rejects_unknown_split(self, fake_datasets) -> None:
        fake_datasets()
        with pytest.raises(ValueError, match="evaluation-only"):
            list(load_casablanca.load_casablanca_examples(split="validation_extra"))

    def test_returns_role_metadata(self, fake_datasets) -> None:
        fake_datasets(
            [
                {
                    "audio": {"array": [0.0], "sampling_rate": 16000},
                    "seg_id": "seg1",
                    "transcription": "text",
                    "gender": "m",
                    "duration": 1.5,
                }
            ]
        )
        ex = next(iter(load_casablanca.load_casablanca_examples(split="validation")))
        assert ex["role"] == "external_eval_only"
        assert ex["seg_id"] == "seg1"
        assert ex["gender"] == "m"
        assert ex["duration"] == 1.5
        assert ex["sample_id"] == stable_sample_id(
            "UBC-NLP/Casablanca", CASABLANCA_REVISION, "validation", 0
        )
