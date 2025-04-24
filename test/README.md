# Kontxt Server Tests

This directory contains unit tests for the Kontxt MCP Server.

## Running the Tests

To run all tests:

```bash
python -m test.test_kontxt_server
```

## Test Coverage

The tests verify the following functionality:

1. **Server Initialization** - Tests that the server initializes correctly with the provided configuration.
2. **get_codebase_context Tool** - Tests the main functionality of the MCP server by simulating a client request to get codebase context.
3. **list_repository_structure Tool** - Tests the internal tool that lists the repository structure.

## Mock Usage

The tests use `unittest.mock` to mock external dependencies:
- The Google Generative AI API (Gemini) is mocked to avoid actual API calls
- Subprocess calls are mocked to avoid actual shell command execution

This approach allows testing the internal logic without relying on external services.