# ЛР2 · Оптимізація моделі: конвертація в ONNX та квантизація

Повний текст інструкції — у файлі `ЛР2_Методичні_вказівки.docx` поруч
із цим репозиторієм. Цей README — лише технічна навігація по коду.

Робота ПРОДОВЖУЄ сервіс із ЛР1: `app/schemas.py`, `app/main.py`
(ендпоінти, lifespan, middleware) і базовий `SentimentClassifier`
переносяться без змін. Новий код — лише те, що додає ONNX і
квантизацію.

Два рівні:

- **Базовий (обов'язковий)** — кроки 1–7: експорт у ONNX, `ONNXSentimentClassifier`,
  перемикання бекенда через Settings, бенчмарк PyTorch vs ONNX.
  Перевіряється `pytest -m "not advanced"`.
- **Поглиблений (додатково)** — кроки 8–12: динамічна INT8-квантизація
  з апаратно-залежним вибором конфігурації, 3-way бенчмарк, warm-up
  і `OMP_NUM_THREADS` для відтворюваності.
  Перевіряється `pytest -m advanced`.

## Структура

```
app/
  schemas.py    — контракт API: додано поле backend у HealthResponse (не редагувати)
  config.py     — Settings: додано backend, onnx_model_dir, onnx_int8_model_dir (не редагувати)
  model.py      — SentimentClassifier (з ЛР1, готовий) + ONNXSentimentClassifier (TODO: load())
  main.py       — усе з ЛР1 без змін + TODO: create_classifier() (крок 5)
scripts/
  benchmark.py  — TODO: measure_latency(), measure_accuracy() (крок 6)
  quantize.py   — TODO: pick_quantization_config() (крок 9, advanced)
tests/
  test_api.py, test_batch.py, test_concurrency.py   — з ЛР1, без змін
  test_backend_switch.py   — тести фабрики create_classifier()
  test_onnx_classifier.py  — тести ONNXSentimentClassifier (фейкові optimum/transformers)
  test_quantization.py      [advanced] — тести pick_quantization_config()
  test_real_model.py        — тест на реальній моделі (вручну)
requirements.txt
pytest.ini
```

## Встановлення

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Порядок виконання

**Базовий рівень**

1. `optimum-cli export onnx --model distilbert-base-uncased-finetuned-sst-2-english models/onnx-fp32`
2. `pytest -m "not advanced" -v` — тести test_onnx_classifier.py і test_backend_switch.py мають впасти з NotImplementedError.
3. Реалізувати `ONNXSentimentClassifier.load()` в `app/model.py`.
4. Реалізувати `create_classifier()` в `app/main.py`.
5. `pytest -m "not advanced" -v` — усе має позеленіти.
6. Реалізувати `measure_latency()`/`measure_accuracy()` в `scripts/benchmark.py`.
7. `python scripts/benchmark.py pytorch onnx` — заповнити таблицю звіту.

**Поглиблений рівень**

8. Реалізувати `pick_quantization_config()` в `scripts/quantize.py`.
9. `pytest tests/test_quantization.py -v` — перевірити логіку вибору конфігурації.
10. `python scripts/quantize.py` — отримати `models/onnx-int8/`.
11. `OMP_NUM_THREADS=1 python scripts/benchmark.py pytorch onnx onnx-int8` — 3-way порівняння.
12. Перемкнути сервіс на кожен бекенд через `OK11_BACKEND=onnx uvicorn app.main:app` і перевірити `/health`.

## Типові проблеми

Див. розділ «Типові помилки» методички.
