# Contributing to AdaptiveRL

We welcome contributions to the AdaptiveRL educational drone demonstration project!

## Development Guidelines

1. **Virtual Environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[rl,dev]"
   ```

2. **Code Quality**:
   - Format: `ruff format src/ tests/`
   - Lint: `ruff check src/ tests/`
   - Static Typecheck: `mypy src/`
   - Test Suite: `pytest -v tests/`

3. **Core Philosophy**:
   - Keep the repository focused on the 3D drone RL demonstration story.
   - Do not re-introduce speculative research frameworks, GUI wrappers, or multi-domain environments.
