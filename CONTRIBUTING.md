# Contributing

Thank you for your interest in SPADE.

## Issues

Open an issue for bugs, questions, documentation problems, or reproduction gaps. Please include the Python version, dependency versions, command used, and the full traceback when reporting errors.

## Pull requests

Use pull requests for fixes and small improvements. Keep changes focused and explain the motivation in the PR description.

Before opening a PR, run:

```bash
python -m pytest
```

If the change affects the project page, also preview it locally:

```bash
python3 -m http.server 8000 -d docs
```

Do not commit large datasets, benchmark dumps, checkpoints, generated experiment outputs, or private credentials.
