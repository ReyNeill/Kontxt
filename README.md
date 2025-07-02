[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/61b0cc9b-59e5-4fd6-8bf8-aa164f5d0006)
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
- `--gemini-model NAME`: Specific Gemini model to use (default: 'gemini-2.0-flash').
- `--transport {stdio,sse}`: Transport protocol to use (default: sse).
- `--host HOST`: Host address for the SSE server (default: 127.0.0.1).
- `--port PORT`: Port for the SSE server (default: 8080).

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
