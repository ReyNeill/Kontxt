# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run/Test Commands
- Setup environment: `python -m venv venv && source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run server: `python kontxt_server.py --repo-path /path/to/codebase`
- Run with stdio transport: `python kontxt_server.py --repo-path /path/to/codebase --transport stdio`
- Run tests: `python -m test.test_kontxt_server`

## Code Style Guidelines
- Follow PEP 8 conventions for Python code
- Use type hints (from typing module) for function parameters and return values
- Organize imports: stdlib first, then third-party, then local modules
- Use f-strings for string formatting
- Handle exceptions with specific exception types
- Use descriptive variable and function names
- Log important operations and errors with the logging module
- Document classes and functions with docstrings