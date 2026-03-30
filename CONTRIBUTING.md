# Contributing to NeuraBreak

Thanks for your interest! Contributions are welcome — code, bug reports, dataset frames.

## Getting Started

1. Fork the repo and clone your fork
2. Set up the dev environment:
   ```bash
   uv sync --extra dev
   uv run pre-commit install
   ```
3. Create a branch: `git checkout -b my-feature`
4. Make your changes
5. Run the test suite: `uv run pytest tests/ -v`
6. Commit and push, then open a PR

## Code Style

- **Python 3.11+** — use built-in `tomllib`, `match`, etc. where they make sense
- `ruff` for linting and formatting — run `uv run ruff check src/ tests/` before committing
- Pre-commit hooks enforce this automatically after `pre-commit install`
- Type hints on all public functions; mypy should pass cleanly
- Comments explain *why*, not *what*

## Testing

- Put unit tests in `tests/unit/`, integration tests in `tests/integration/`
- Tests must not require a camera, display, or network connection
- Mock heavy dependencies (`QApplication`, `cv2.VideoCapture`, etc.) when needed
- Aim for real assertions, not just smoke tests

## Submitting a PR

- Keep PRs focused — one feature or fix per PR
- Update or add tests for any changed behaviour
- If you're adding a new UI component, describe the interaction in the PR description
- Fill out the PR template

## Dataset Contributions

If you'd like to improve the model, you can collect and submit annotated frames. See the [dataset contribution template](.github/ISSUE_TEMPLATE/dataset_contribution.md) and [training/README.md](training/README.md) for details.

All contributed frames must:
- Be captured by you (no third-party footage)
- Not contain identifiable faces unless you are the subject
- Be submitted under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence

## Licence

By contributing, you agree that your contributions will be licensed under the [MIT Licence](LICENSE).
