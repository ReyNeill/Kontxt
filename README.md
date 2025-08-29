[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/61b0cc9b-59e5-4fd6-8bf8-aa164f5d0006)

### Marketplace badges
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/reyneill-kontxt-badge.png)](https://mseep.ai/app/reyneill-kontxt)

# Kontxt MCP Server

A Model Context Protocol (MCP) server that tries to solve condebase indexing (until agents can).

## Features

- Connects to a user-specified local code repository.
- Provides the (`get_codebase_context`) tool for AI clients (like Cursor, Claude Desktop).
- Uses Gemini 2.0 Flash's 1M input window internally to analyze the codebase and generate context based on the user's client querry.
- Flash itself can use internal tools (`list_repository_structure`, `read_files`, `grep_codebase`) to understand the code.
- Supports both SSE (recommended) and stdio transport protocols.
- Supports user-attached files/docs/context from client's queries for more targeted analysis.
- Tracks token usage and provides detailed analysis of API consumption.
- User-configurable token limit for context generation (options: 500k, 800k, or 1M tokens; default: 800k).

## Setup

1.  **Clone/Download:** Get the server code.
2.  **Create Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install `tree`:** Ensure the `tree` command is available on your system.
    - macOS: `brew install tree`
    - Debian/Ubuntu: `sudo apt update && sudo apt install tree`
    - Windows: Requires installing a port or using WSL.
5.  **Configure API Key:**
    - Copy `.env.example` to `.env`.
    - Edit `.env` and add your Google Gemini API Key:
      ```
      GEMINI_API_KEY="YOUR_ACTUAL_API_KEY"
      ```
    - Alternatively, you can provide the key via the `--gemini-api-key` command-line argument.

## Running as a Standalone Server (Recommended)

By default, the server runs in SSE mode, which allows you to:
- Start the server independently
- Connect from multiple clients
- Keep it running while restarting clients

Run the server:

```bash
python kontxt_server.py --repo-path /path/to/your/codebase
```

PS: you can use ```pwd``` to list the project path

The server will start on `http://127.0.0.1:8080/sse` by default.

For additional options:
```bash
python kontxt_server.py --repo-path /path/to/your/codebase --host 0.0.0.0 --port 6900
```

### Shutting Down the Server

The server can be stopped by pressing `Ctrl+C` in the terminal where it's running. The server will attempt to close gracefully with a 3-second timeout.

## Connecting to the Server from client (Cursor example)

Once your server is running, you can connect Cursor to it by editing your `~/.cursor/mcp.json` file:

```json
{
  "mcpServers": {
    "kontxt-server": {
      "serverType": "sse",
      "url": "http://localhost:8080/sse"
    }
  }
}
```

PS: remember to always refresh the MCP server on Cursor Settings or other client to connect to the MCP via sse

## Alternative: Running with stdio Transport

If you prefer to have the client start and manage the server process:

```bash
python kontxt_server.py --repo-path /path/to/your/codebase --transport stdio
```

For this mode, configure your `~/.cursor/mcp.json` file like this:

```json
{
  "mcpServers": {
    "kontxt-server": {
      "serverType": "stdio",
      "command": "python",
      "args": ["/absolute/path/to/kontxt_server.py", "--repo-path", "/absolute/path/to/your/codebase", "--transport", "stdio"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Command Line Arguments

- `--repo-path PATH`: **Required**. Absolute path to the local code repository to analyze.
- `--gemini-api-key KEY`: Google Gemini API Key (overrides `.env` if provided).
- `--token-threshold NUM`: Target maximum token count for the context. Allowed values are:
  - 500000
  - 800000 (default)
  - 1000000
- `--gemini-model NAME`: Specific Gemini model to use (default: `models/gemini-2.5-flash-preview-04-17`).
- `--tokenizer-model NAME`: Hugging Face tokenizer id for token estimation (default: `google/gemma-7b`; override via `KONTXT_TOKENIZER_MODEL`).
- `--transport {stdio,sse}`: Transport protocol to use (default: sse).
- `--host HOST`: Host address for the SSE server (default: 127.0.0.1).
- `--port PORT`: Port for the SSE server (default: 8080).
- `--cors-origins ORIGINS`: Comma-separated list of allowed CORS origins. If omitted, defaults to loopback only.
- `--cors-credentials`: Allow credentials for CORS (disabled by default).

### CORS Configuration

For security, wildcard CORS is not used. By default, only loopback origins are allowed:

- `http://127.0.0.1`, `http://localhost`, and the bound `host:port`.

To allow specific web clients during development, pass explicit origins or use an env var:

```bash
python kontxt_server.py \
  --repo-path /path/to/your/codebase \
  --cors-origins http://localhost:3000,http://127.0.0.1:5173

# or via environment variable
KONTXT_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:5173" \
python kontxt_server.py --repo-path /path/to/your/codebase
```

Notes:
- Allowed methods: `GET`, `OPTIONS`. Headers: all. Credentials: off unless `--cors-credentials` is set.

## Tokenizer (Gemma) Access & Auto-Recovery

This server uses the `google/gemma-7b` tokenizer to estimate tokens. The model is gated by Google on Hugging Face.

What happens if you don't have access yet:
- On startup, if the tokenizer cannot be downloaded, the server logs a clear message and auto-opens: https://huggingface.co/google/gemma-7b
- The server keeps running using a heuristic token estimator (does not crash).
- It periodically retries loading the tokenizer; once you gain access, it switches automatically (no restart needed).

How to gain access (free, ~2 minutes):
1) Visit https://huggingface.co/google/gemma-7b and log in (create an account if needed).
2) Accept Google’s terms on the model page.
3) If running headless/CI or a container, authenticate the environment: `huggingface-cli login` (or set `HF_TOKEN`).

Configuration:
- `--tokenizer-model` or `KONTXT_TOKENIZER_MODEL`: use a different HF tokenizer id if desired.
- `KONTXT_TOKENIZER_RELOAD_INTERVAL` (seconds, default 60): how often the server re-attempts tokenizer loading.

### Basic Usage

Example queries:
- "What's this codebase about"
- "How does the authentication system work?"
- "Explain the data flow in the application"

PS: you can further specify the agent to use the MCP tool if it's not using it: "What is the last word of the third codeblock of the auth file? Use the MCP tool available."

### Context Attachment

Your referenced files/context in your queries are included as context for analysis:

- "Explain how this file works: @kontxt_server.py"
- "Find all files that interact with @user_model.py"
- "Compare the implementation of @file1.js and @file2.js"


The server will mention these files to Gemini but will NOT automatically read or include their contents. Instead, Gemini will decide which files to read using its tools based on the query context.

This approach allows Gemini to only read files that are actually needed and prevents the context from being bloated with irrelevant file content.

## Token Usage Tracking

The server tracks token usage across different operations:
- Repository structure listing
- File reading
- Grep searches
- Attached files from user queries
- Generated responses

This information is logged during operation, helping you monitor API usage and optimize your queries.

PD: want the tool to improve? PR's are open.
