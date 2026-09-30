# Contributing

Bug reports and focused improvements are welcome. Please include a small synthetic checkpoint fixture when reporting an index or SafeTensors parsing issue; never attach private weights, credentials, or production model artifacts.

Before opening a pull request, run:

```sh
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy
```

Keep format claims precise. In particular, do not deserialize `.bin` files or read SafeTensors payloads as part of an audit.
