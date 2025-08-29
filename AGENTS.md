# Repository Guidelines

## Project Structure & Module Organization
- `kontxt_server.py`: Main MCP server (FastMCP + FastAPI/SSE) and CLI entry.
- `test/`: Unit tests (unittest), e.g., `test/test_kontxt_server.py`.
- `requirements.txt`: Python dependencies.
- `.env.example` / `.env`: Configuration (e.g., `GEMINI_API_KEY`). Never commit secrets.

## Build, Test, and Development Commands
```bash
# Create and activate a virtualenv
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server (SSE transport)
python kontxt_server.py --repo-path /path/to/repo

# Run server (stdio transport)
python kontxt_server.py --repo-path /abs/path --transport stdio

# Run tests
python -m test.test_kontxt_server
# or
python test/test_kontxt_server.py
```

## Coding Style & Naming Conventions
- Follow PEP 8, 4‑space indentation; add type hints where practical.
- Naming: files/modules `snake_case.py`; functions/vars `snake_case`; classes `CapWords`; constants `UPPER_SNAKE_CASE`.
- Docstrings: concise triple‑quoted summaries (purpose, params, returns).
- Logging: use `logging` configured in `kontxt_server.py`; avoid `print`.

## Testing Guidelines
- Framework: `unittest` with `unittest.mock` for external APIs (Gemini) and subprocess.
- Scope: add tests for new tools/endpoints and argument parsing; keep network/API calls mocked.
- Naming: place tests under `test/`, files `test_*.py`, methods `test_*`.
- Ensure tests pass locally without real API keys.

## Commit & Pull Request Guidelines
- Commits: prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`). Examples in history: `feat: update gemini model…`, `docs: update README…`.
- PRs should include: summary, reasoning, commands to run, linked issues, and test output. For behavior changes, include example invocation.

## Security & Configuration Tips
- Secrets: set `GEMINI_API_KEY` in `.env` or via `--gemini-api-key`. Do not commit `.env`.
- Be mindful of token usage; avoid reading large files unnecessarily.
- Ensure `tree` is installed if you rely on repo structure listings.

## Architecture Overview
- Core server exposes FastMCP tools: `list_repository_structure`, `read_files`, `grep_codebase`, with token tracking via `TokenTracker` and a Gemini client.
- Transport: SSE by default on `/sse`; stdio optional for client‑managed startup.

