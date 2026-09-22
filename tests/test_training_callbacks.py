from types import SimpleNamespace

import numpy as np

from masricx.training.callbacks import compute_wer


class _Processor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def batch_decode(self, values: np.ndarray, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens
        mapping = {
            (1, 2, 0): "افتح account",
            (1, 3, 0): "افتح الحساب",
            (4, 5, 0): "رقم order",
            (4, 6, 0): "رقم order",
        }
        return [mapping[tuple(row)] for row in values]


def test_compute_wer_includes_micro_averaged_english_recall() -> None:
    prediction = SimpleNamespace(
        predictions=np.array([[1, 3, 0], [4, 6, 0]]),
        label_ids=np.array([[1, 2, -100], [4, 5, -100]]),
    )

    metrics = compute_wer(_Processor(), prediction)

    assert metrics == {"wer": 0.25, "english_term_recall": 0.5}
