# Repository Guidelines

## Project Structure & Module Organization
- `kontxt_server.py`: Main MCP server (FastMCP + FastAPI/SSE) and CLI entry.
- `test/`: Unit tests (unittest) e.g., `test/test_kontxt_server.py`.
- `requirements.txt`: Python dependencies. Use a virtualenv.
- `.env.example` / `.env`: Configuration (e.g., `GEMINI_API_KEY`). Never commit secrets.

## Build, Test, and Development Commands
- Create env: `python -m venv venv && source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run server (SSE): `python kontxt_server.py --repo-path /path/to/repo`
- Run server (stdio): `python kontxt_server.py --repo-path /abs/path --transport stdio`
- Run tests: `python -m test.test_kontxt_server` (or `python test/test_kontxt_server.py`)

## Coding Style & Naming Conventions
- Follow PEP 8, 4‑space indentation, type hints where practical.
- Naming: modules/files `snake_case.py`, functions/vars `snake_case`, classes `CapWords`, constants `UPPER_SNAKE_CASE`.
- Docstrings: triple‑quoted, concise purpose + params/returns.
- Logging: use `logging` (already configured in `kontxt_server.py`); avoid `print`.

## Testing Guidelines
- Framework: `unittest` with `unittest.mock` for external APIs (Gemini) and subprocess.
- Location: place tests under `test/`, name files `test_*.py` and methods `test_*`.
- Scope: add tests for new tools/endpoints and argument parsing; keep network/API calls mocked.
- Run locally before PR; ensure tests pass without real API keys.

## Commit & Pull Request Guidelines
- Commit style: prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`). Examples in history: `feat: update gemini model…`, `docs: update README…`.
- PRs should include: brief description, reasoning, commands to run, linked issues, and test output. For behavior changes, show example invocation.

## Security & Configuration Tips
- Secrets: set `GEMINI_API_KEY` in `.env` or pass `--gemini-api-key`. Do not commit `.env`.
- Dependencies: ensure `tree` is installed for repo structure listing.
- Token usage: be mindful of tokenizer calls; avoid reading large files unnecessarily.

## Architecture Overview
- Core server class integrates FastMCP tools: `list_repository_structure`, `read_files`, `grep_codebase`, with token tracking via `TokenTracker` and Gemini client usage.
- Transport: SSE by default (`/sse`); stdio optional for client‑managed startup.

