# Contributing to bielik-one

Thank you for helping with bielik-one, the serving runtime for the Bielik One decision models. Bug reports, fixes, documentation improvements and new features are welcome.

## Issues

Open a GitHub issue for bugs, questions and feature ideas. For a bug, include:

- the bielik-one version (`pip show bielik-one`), vLLM version, GPU and driver;
- the package (Hugging Face repo id and revision, or adapter/merged), and the `bielik-one backend|frontend` command with its flags;
- a minimal `/v1/systemone` request that shows the problem (without confidential data), the response, and the frontend/backend log excerpt.

Questions about model quality (a wrong decision on your data) are welcome too; please include the state, question and options.

## Security

Do not open a public issue for a vulnerability (for example a way to make the public API pass internal decision flags to the backend, or to inject control tokens). Report it privately via GitHub's "Report a vulnerability" on the Security tab of this repository.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request focused on one change; describe what it fixes and how you tested it.
3. Add or update tests in `tests/` for behaviour changes. Tests use synthetic backends and need no GPU; GitHub Actions runs them on every pull request:

   ```bash
   pip install -e ".[test]" && pytest -q
   ```

4. Update `README.md` when you change flags, endpoints, limits or defaults.

Changes to prompt rendering (`bielik_one/prompting/`), decision reads (`bielik_one/api/remote_vllm.py`, `merged_vllm.py`) or the vLLM plugin (`vllm-heads-plugin/`) can silently change model outputs. The served prompts must stay token-identical to the ones the models were trained and evaluated on. For such changes, please say whether you checked decisions against an unchanged server on a real GPU (for example a few hundred validation-style questions per question type) and what agreement you saw. Maintainers may run this check before merging.

Dependency pins (`vllm`, `torch`, `transformers` in `pyproject.toml`) follow the vLLM release the runtime is tested with; upgrading them needs the same GPU check.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0, the license of this repository (see `LICENSE`).
