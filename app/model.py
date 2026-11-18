"""
Обгортки над моделлю класифікації тональності.

SentimentClassifier — перенесено з ЛР1 БЕЗ ЗМІН (готовий, не редагувати).
ONNXSentimentClassifier — новий у ЛР2 клас. Зверніть увагу: він
успадковує predict(), predict_batch() та is_ready від SentimentClassifier
без жодного перевизначення — ваше завдання (крок 4) обмежується
методом load().
"""

from __future__ import annotations

from transformers import pipeline


class SentimentClassifier:
    """PyTorch-версія — еталонний бекенд, з яким порівнюються інші (з ЛР1)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None

    def load(self) -> None:
        self._pipeline = pipeline("sentiment-analysis", model=self.model_name)

    def predict(self, text: str) -> tuple[str, float]:
        result = self._pipeline(text)[0]
        return result["label"], float(result["score"])

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        results = self._pipeline(texts)
        return [(r["label"], float(r["score"])) for r in results]

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None


class ONNXSentimentClassifier(SentimentClassifier):
    """
    ONNX-версія класифікатора тональності тексту.
    Підтримує FP32 та INT8 моделі.
    """

    def __init__(self, model_name: str, model_dir: str) -> None:
        super().__init__(model_name)
        self.model_dir = model_dir

    def load(self) -> None:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        model = ORTModelForSequenceClassification.from_pretrained(
            self.model_dir
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir
        )

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )


def load(self) -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    model = ORTModelForSequenceClassification.from_pretrained(
        self.model_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        self.model_dir
    )

    self._pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer
    )
