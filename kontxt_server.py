import asyncio
import argparse
import os
import sys
import subprocess
import json
import logging
import uuid
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union, Tuple
import aiofiles
from transformers import AutoTokenizer
import heapq

# FastMCP instead of low-level Server
from mcp.server.fastmcp import FastMCP, Context
# Import ClientSession from top-level mcp package
from mcp import ClientSession
import mcp.types as types
import mcp.server.stdio as mcp_stdio

# FastAPI for SSE support
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# gemini imports
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- token tracking ---
class TokenTracker:
    """Helper class to track token usage in tools and responses"""
    
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.reset()
        # using gemma tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    
    def reset(self):
        """Reset all tracked token counts"""
        self.total_estimated_tokens = 0
        self.tool_tokens = {
            "list_repository_structure": 0,
            "read_files": 0,
            "grep_codebase": 0,
            "referred_file": 0
        }
        self.response_tokens = 0
    
    def estimate_tokens(self, text: str) -> int:
        """Token estimation using the Gemma tokenizer (SentencePiece-based)"""
        if not text:
            return 0
        # tokenizer's encode method to count tokens
        return len(self.tokenizer.encode(text))
    
    def track_tool_usage(self, tool_name: str, content: str):
        """Track token usage for a specific tool"""
        tokens = self.estimate_tokens(content)
        self.tool_tokens[tool_name] = self.tool_tokens.get(tool_name, 0) + tokens
        self.total_estimated_tokens += tokens
        
        # log token usage and percentage of threshold
        percentage = (self.total_estimated_tokens / self.threshold) * 100 if self.threshold > 0 else 0
        logger.info(f"Token tracking: {tool_name} used ~{tokens} tokens (total: ~{self.total_estimated_tokens}, {percentage:.1f}% of threshold)")
    
    def track_response(self, response_text: str):
        """Track tokens in the final response"""
        self.response_tokens = self.estimate_tokens(response_text)
        total_with_response = self.total_estimated_tokens + self.response_tokens
        percentage = (total_with_response / self.threshold) * 100 if self.threshold > 0 else 0
        logger.info(f"Token tracking: Final response ~{self.response_tokens} tokens (grand total: ~{total_with_response}, {percentage:.1f}% of threshold)")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Return a summary of token usage"""
        return {
            "total_estimated_tokens": self.total_estimated_tokens,
            "tool_usage": self.tool_tokens,
            "response_tokens": self.response_tokens,
            "threshold": self.threshold,
            "percentage_used": (self.total_estimated_tokens + self.response_tokens) / self.threshold * 100 if self.threshold > 0 else 0
        }

# --- response analysis ---
def analyze_response_structure(text: str) -> Dict[str, Any]:
    """Analyze the structure of a Gemini response and extract key components"""
    
    # check for code blocks
    code_blocks_count = text.count("```")
    code_blocks = code_blocks_count // 2  # divide by 2 since each block has opening and closing
    
    # check for headings (Markdown style)
    heading_matches = re.findall(r'^#+\s+.*$', text, re.MULTILINE)
    headings = len(heading_matches)
    
    # get a list of heading titles for a better overview
    heading_titles = []
    for match in heading_matches:
        # strip the # symbols and whitespace
        title = re.sub(r'^#+\s+', '', match).strip()
        if title:
            heading_titles.append(title)
    
    # check for bullet points/lists
    list_items = len(re.findall(r'^\s*[-*]\s+.*$', text, re.MULTILINE))
    
    # estimate paragraphs (text blocks separated by blank lines)
    paragraphs = len(re.findall(r'\n\s*\n', text)) + 1
    
    # count total lines
    lines = text.count('\n') + 1
    
    return {
        "code_blocks": code_blocks,
        "headings": headings,
        "heading_titles": heading_titles,
        "list_items": list_items,
        "paragraphs": paragraphs,
        "lines": lines,
        "chars": len(text)
    }

# --- argument parsing ---
def parse_arguments():
    logger.info("Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Kontxt MCP Server: Provides AI-driven codebase context.")
    parser.add_argument("--repo-path", type=str, required=True,
                        help="Absolute path to the local code repository to analyze.")
    parser.add_argument("--gemini-api-key", type=str, default=None,
                        help="Google Gemini API Key (overrides .env file if provided).")
    parser.add_argument("--token-threshold", type=int, default=800000,
                        help="Target maximum token count for the generated context (default: 800000).")
    parser.add_argument("--gemini-model", type=str, default='gemini-2.0-flash',
                        help="The specific Gemini model to use (default: 'gemini-2.0-flash').")
    # host and port for SSE server
    parser.add_argument("--host", type=str, default="127.0.0.1", help="host for the SSE server (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8080, help="port for the SSE server (default: 8080).")
    # transport option
    parser.add_argument("--transport", type=str, choices=["stdio", "sse"], default="sse",
                      help="Transport protocol to use (stdio or sse, default: sse).")
    args = parser.parse_args()
    logger.info(f"Arguments parsed: repo_path={args.repo_path}, model={args.gemini_model}, transport={args.transport}")
    return args

# --- utility functions ---
def check_tree_command():
    """Checks if the 'tree' command is available."""
    logger.info("Checking for 'tree' command availability")
    try:
        subprocess.run(["tree", "--version"], check=True, capture_output=True)
        logger.info("'tree' command found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("'tree' command not found. Please install it (e.g., 'brew install tree' or 'sudo apt install tree') and ensure it's in your PATH.")
        return False

def extract_file_content_from_query(query: str) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Extracts file content embedded in a query and returns the cleaned query, extracted content,
    and any @file references.
    
    Detects patterns like:
    1. Code blocks with path:
    ```path=path/to/file.ext
    file content here
    ```
    
    2. Documentation sections:
    Document Name: document-name
    Document URL: url
    Document content:
    content here
    ____
    
    3. @file references in the query text
    
    Args:
        query: The original query text
    
    Returns:
        Tuple of (cleaned_query, extracted_files_dict, at_file_references)
    """
    extracted_files = {}
    at_file_references = []
    
    # pattern 1: match embedded file content with path info
    code_pattern = r'```path=([^\n,]+)(?:, lines=\d+-\d+)?\n([\s\S]*?)```'
    code_matches = re.findall(code_pattern, query)
    
    # extract files and their content from code blocks
    for file_path, content in code_matches:
        filename = file_path.split('/')[-1]  # extract filename from path
        extracted_files[filename] = content
    
    # pattern 2: match documentation sections with Document Name, URL and content
    doc_pattern = r'Document Name:\s*([^\n]+)\s*\nDocument URL:[^\n]*\s*\nDocument content:\s*\n([\s\S]*?)(?:____|$)'
    doc_matches = re.findall(doc_pattern, query)
    
    # extract files and their content from documentation sections
    for doc_name, content in doc_matches:
        # clean up the doc name to use as filename
        filename = doc_name.strip().replace(' ', '_') + ".md"
        extracted_files[filename] = content.strip()
    
    # pattern 3: extract @file references from the query
    at_file_pattern = r'@([^\s,;]+)'
    at_matches = re.findall(at_file_pattern, query)
    at_file_references.extend(at_matches)
    
    # check if we have any matches
    if not code_matches and not doc_matches and not at_matches:
        return query, {}, []
    
    # remove the file content blocks from the query
    cleaned_query = re.sub(code_pattern, '', query)
    
    # remove documentation sections from the query
    # use a simplified pattern that captures the entire documentation section
    doc_section_pattern = r'Document Name:[\s\S]*?(?:____|$)'
    cleaned_query = re.sub(doc_section_pattern, '', cleaned_query)
    
    # remove any double newlines and clean up
    cleaned_query = re.sub(r'\n\n+', '\n\n', cleaned_query)
    cleaned_query = re.sub(r'-------\s*\n\n+', '', cleaned_query)  # remove section dividers
    cleaned_query = re.sub(r'## Potentially Relevant Documentation:\s*\n\n+', '', cleaned_query)  # remove section headers
    
    # trim whitespace
    cleaned_query = cleaned_query.strip()
    
    return cleaned_query, extracted_files, at_file_references

# --- main server class ---
class KontxtMcpServer:
    def __init__(self, repo_path: Path, gemini_api_key: str, token_threshold: int, gemini_model: str):
        logger.info(f"Initializing KontxtMcpServer with model={gemini_model}")
        # create a FastMCP server instance
        self.mcp = FastMCP("KontxtServer")
        logger.info("Created FastMCP server instance")
        
        if not repo_path.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {repo_path}")
        self.repo_path = repo_path
        self.gemini_api_key = gemini_api_key
        self.token_threshold = token_threshold
        self.gemini_model_name = gemini_model
        self.gemini_client: Optional[genai.GenerativeModel] = None
        
        # initialize token tracker
        self.token_tracker = TokenTracker(token_threshold)
        logger.info(f"Initialized token tracker with threshold of {token_threshold} tokens")

        # define Gemini tool schemas using Gemini types
        logger.info("Defining Gemini tool schemas")
        self.gemini_tool_schemas = [
            FunctionDeclaration(
                name="list_repository_structure",
                description=f"Lists the directory structure of the codebase at {self.repo_path}, ignoring common build/dependency directories.",
                parameters={}
            ),
            FunctionDeclaration(
                name="read_files",
                description="Reads the content of specified files within the codebase.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "paths": {
                            "type_": "ARRAY",
                            "description": "List of file paths relative to the repository root to read.",
                            "items": {"type_": "STRING"}
                        }
                    },
                    "required": ["paths"]
                }
            ),
            FunctionDeclaration(
                name="grep_codebase",
                description="Searches for a regex pattern within files in the codebase.",
                parameters={
                    "type_": "OBJECT",
                    "properties": {
                        "pattern": {
                            "type_": "STRING",
                            "description": "The regex pattern to search for (use standard regex syntax)."
                        },
                    },
                    "required": ["pattern"]
                }
            )
        ]
        self.gemini_tools_config = Tool(function_declarations=self.gemini_tool_schemas)
        logger.info("Defined Gemini tool schemas and tool configuration")

        # initialize Gemini Client here, as it's needed before MCP initialize is called by client
        self._initialize_gemini_client()
        # register MCP handlers
        self._register_handlers()
        logger.info("KontxtMcpServer initialization complete")
        
    def _initialize_gemini_client(self):
        logger.info(f"Initializing Gemini client with model: {self.gemini_model_name}")
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai.GenerativeModel(
                self.gemini_model_name,
                tools=self.gemini_tools_config
            )
            logger.info(f"Successfully initialized Gemini client with model: {self.gemini_model_name}")
        except ImportError:
            logger.error("'google-generativeai' library not found. Please install it: pip install google-generativeai")
            raise RuntimeError("Missing google-generativeai dependency")
        except Exception as e:
            logger.error(f"Failed to configure or initialize Gemini API: {e}")
            # allow server to start, but tool calls will fail
            self.gemini_client = None
            logger.warning("Gemini client failed to initialize. Tool calls will not work.")

    def _get_prioritized_file_list(self):
        """
        Returns a prioritized list of files for context inclusion.
        Priority: universal important files (README, LICENSE, etc.), docs, config, API/auth/schema/types, .cursor, CLAUDE.md, main entry points, then by size (smallest first), then others.
        """
        important = []
        docs = []
        config = []
        api = []
        auth = []
        schemas = []
        types = []
        cursor = []
        claude = []
        main_entry = []
        others = []
        for root, dirs, files in os.walk(self.repo_path):
            rel_root = os.path.relpath(root, self.repo_path)
            # skip common build/dependency dirs
            if any(part in rel_root for part in ['.git', 'venv', '__pycache__', 'node_modules', 'dist', 'build']):
                continue
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), self.repo_path)
                lower = f.lower()
                # universally important files
                if lower.startswith(('readme', 'license', 'contributing', 'changelog', 'security', 'auth')) or lower in {'claude.md'}:
                    important.append(rel_path)
                # docs
                elif rel_root.startswith('docs') or lower.endswith('.md'):
                    docs.append(rel_path)
                # config
                elif lower.startswith('config') or lower.startswith('.env') or rel_root.startswith('config'):
                    config.append(rel_path)
                # .cursor
                elif '.cursor' in rel_root or lower.startswith('.cursor'):
                    cursor.append(rel_path)
                # claude
                elif 'claude' in lower:
                    claude.append(rel_path)
                # API endpoints
                elif 'api' in rel_root or 'routes' in rel_root or 'endpoint' in rel_root or 'api' in lower or 'route' in lower or 'endpoint' in lower:
                    api.append(rel_path)
                # auth
                elif 'auth' in rel_root or 'auth' in lower:
                    auth.append(rel_path)
                # schemas
                elif 'schema' in rel_root or 'schema' in lower or 'model' in lower:
                    schemas.append(rel_path)
                # types
                elif 'types' in rel_root or 'types' in lower:
                    types.append(rel_path)
                # main entry points
                elif lower.startswith(('main', 'index', 'server', 'app')):
                    main_entry.append(rel_path)
                # TypeScript/JavaScript/Python/other common extensions
                elif lower.endswith(('.ts', '.tsx', '.js', '.jsx', '.py', '.go', '.rs', '.java', '.c', '.cpp', '.cs', '.rb', '.php', '.swift', '.kt')):
                    others.append(rel_path)
                else:
                    others.append(rel_path)
        # sort others by file size (smallest first)
        others = sorted(others, key=lambda p: os.path.getsize(os.path.join(self.repo_path, p)))
        # compose the final prioritized list
        return (
            important + docs + config + api + auth + schemas + types + cursor + claude + main_entry + others
        )

    def _assemble_max_context(self, token_limit, base_prompt_tokens, request_id=None):
        """
        Iteratively add file contents to maximize token usage up to the cap.
        Returns: (context_str, included_files, stats)
        """
        files = self._get_prioritized_file_list()
        included = []
        context = ""
        tokens_used = base_prompt_tokens
        stats = []
        for rel_path in files:
            full_path = os.path.join(self.repo_path, rel_path)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(32_000)  # cap per file for safety
            except Exception:
                continue
            file_section = f"\n\n# File: {rel_path}\n\n{content}\n"
            file_tokens = self.token_tracker.estimate_tokens(file_section)
            if tokens_used + file_tokens > token_limit:
                if request_id:
                    logger.info(f"[{request_id}] Token cap reached: {tokens_used} tokens used, next file '{rel_path}' would add {file_tokens} tokens (limit: {token_limit})")
                break
            context += file_section
            included.append(rel_path)
            tokens_used += file_tokens
            stats.append((rel_path, file_tokens, tokens_used))
            if request_id:
                logger.info(f"[{request_id}] Included file: {rel_path} ({file_tokens} tokens, cumulative: {tokens_used})")
        if request_id:
            logger.info(f"[{request_id}] Final context assembly: {len(included)} files, {tokens_used} tokens used (limit: {token_limit})")
        return context, included, stats

    def _register_handlers(self):
        """Register MCP request handlers using decorators."""
        logger.info("Registering MCP request handlers")
        # register the get_codebase_context tool with FastMCP
        @self.mcp.tool()
        async def get_codebase_context(request: Dict[str, Any]) -> str:
            """Analyzes the configured codebase using AI (Gemini) based on your query and returns relevant context/documentation.
            
            Args:
                request: Dictionary with the following structure:
                    {
                        "request": {
                            "query": "The user's question or request",
                            "referred_files": {"filename": "location", ...} (optional),
                            "extra_info": "Additional context or information to include" (optional)
                        }
                    }
            
            Usage Notes:
                - The outermost object must have a 'request' key.
                - 'query' is the only required parameter inside 'request'.
                - 'referred_files' should contain filenames as keys and (optionally) their locations as values. DO NOT include file contents in this parameter. Files will be mentioned but NOT automatically read.
                - 'extra_info' can contain any additional context like documentation, file contents, terminal output, git diffs, etc. that may help answer the query
            """
            request_id = str(uuid.uuid4())[:8]  # generates a short request ID for tracking
            
            # extract query, referred files, and extra info from the request
            query = request.get("query", "")
            referred_files = request.get("referred_files", {})
            extra_info = request.get("extra_info", "")
            
            logger.info(f"[{request_id}] Received get_codebase_context tool call with query: {query}")
            if referred_files:
                logger.info(f"[{request_id}] Query references {len(referred_files)} files: {', '.join(referred_files.keys())}")
            if extra_info:
                logger.info(f"[{request_id}] Request includes extra_info ({len(extra_info)} chars)")
            
            # extract embedded file content from the query
            cleaned_query, extracted_files, at_file_references = extract_file_content_from_query(query)
            if extracted_files:
                logger.info(f"[{request_id}] Automatically extracted {len(extracted_files)} file(s) from query")
                # use the cleaned query instead
                query = cleaned_query
                
                # track the extracted files without adding their content (leave the fun to Gemini)
                for filename in extracted_files:
                    logger.info(f"[{request_id}] Noted extracted file reference: {filename}")
            
            # handle @file references by noting them but not reading contents
            if at_file_references:
                logger.info(f"[{request_id}] Found {len(at_file_references)} @file references: {', '.join(at_file_references)}")
                for file_ref in at_file_references:
                    # remove @ prefix if present
                    clean_name = file_ref.lstrip('@')
                    logger.info(f"[{request_id}] Noted @file reference: {clean_name}")
            
            # reset token tracker for this request
            self.token_tracker.reset()
            logger.info(f"[{request_id}] Reset token tracker for new request")
            
            if not query:
                logger.warning(f"[{request_id}] Missing required parameter 'query'")
                return "Error: Missing required parameter 'query'"
            if not self.gemini_client:
                 logger.error(f"[{request_id}] Gemini client not available for tool call.")
                 return "Error: Gemini client not initialized or failed to initialize."

            try:
                logger.info(f"[{request_id}] Processing query with Gemini: {query}")

                chat = self.gemini_client.start_chat(enable_automatic_function_calling=True)
                logger.info(f"[{request_id}] Started Gemini chat session with automatic function calling")

                # prepare the initial prompt
                initial_prompt = (
                    f"You are an expert software developer analyzing the codebase located at {self.repo_path}. "
                    f"The user's query is: '{query}'.\n\n"
                )
                
                # if there are referred files, mention them in the prompt but don't include content
                if referred_files or extracted_files or at_file_references:
                    initial_prompt += "The user has referred to the following files:\n\n"
                    
                    for filename in list(referred_files.keys()) + list(extracted_files.keys()) + [ref.lstrip('@') for ref in at_file_references]:
                        if filename in initial_prompt:
                            continue  # skip duplicates
                        initial_prompt += f"- {filename}\n"
                    
                    initial_prompt += "\nThese files may be relevant to the query. You can use the available tools to read their content if needed. Start by listing the project structure with the list_repository_structure tool.\n\n"
                
                # if there's extra info, add it to the prompt
                if extra_info:
                    extra_info_section = (
                        "### ADDITIONAL CONTEXT PROVIDED BY USER\n\n"
                        f"{extra_info}\n\n"
                        "This additional context may be relevant to the user's query. "
                        "Use it to supplement your analysis of the codebase.\n\n"
                    )
                    initial_prompt += extra_info_section
                    # track token usage for extra info
                    self.token_tracker.track_tool_usage("referred_file", extra_info)
                    logger.info(f"[{request_id}] Added extra_info to prompt ({len(extra_info)} chars)")
                
                # complete the prompt with tool usage instructions
                initial_prompt += (
                    f"Please use the available tools (list_repository_structure, read_files, grep_codebase) "
                    f"to understand the relevant parts of the codebase. Your goal is to generate a comprehensive "
                    f"yet concise context or documentation that directly addresses the user's query. "
                    f"Focus on providing useful information, code snippets, and explanations relevant to the query. "
                    f"Try to keep the total response size reasonable, aiming for clarity over excessive length. "
                    f"Do not just list file contents; synthesize the information."
                )
                
                prompt_additions = []
                if referred_files or extracted_files or at_file_references:
                    prompt_additions.append("file references")
                if extra_info:
                    prompt_additions.append("extra user-provided context")
                    
                prompt_addition_text = ""
                if prompt_additions:
                    prompt_addition_text = " with " + ", ".join(prompt_additions)
                    
                logger.info(f"[{request_id}] Created initial prompt for Gemini{prompt_addition_text}")

                # smart context maximization here
                base_prompt_tokens = self.token_tracker.estimate_tokens(initial_prompt)
                max_tokens = self.token_tracker.threshold
                context_str, included_files, context_stats = self._assemble_max_context(max_tokens, base_prompt_tokens, request_id)
                if included_files:
                    initial_prompt += f"\n\nThe following files have been included for context (up to the token cap):\n"
                    for f in included_files:
                        initial_prompt += f"- {f}\n"
                    initial_prompt += "\n---\n"
                    initial_prompt += context_str
                # log summary of context maximization
                logger.info(f"[{request_id}] Context maximization stats:")
                for rel_path, file_tokens, tokens_used in context_stats:
                    logger.info(f"[{request_id}]   {rel_path}: {file_tokens} tokens (cumulative: {tokens_used})")
                logger.info(f"[{request_id}] Total files included: {len(included_files)}, total tokens for context: {base_prompt_tokens + sum(f[1] for f in context_stats)} (limit: {max_tokens})")
                
                # pass implementations directly to Gemini
                logger.info(f"[{request_id}] Sending message to Gemini with tools")
                response = await chat.send_message_async(
                    initial_prompt,
                    tools=[self.list_repository_structure, self.read_files, self.grep_codebase]
                )
                logger.info(f"[{request_id}] Received response from Gemini")

                # extract final text response
                final_text = ""
                if response.parts:
                    final_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                    logger.info(f"[{request_id}] Extracted text from Gemini response (length: {len(final_text)})")
                else:
                    # handle cases where Gemini might return no parts (e.g., safety settings)
                    logger.warning(f"[{request_id}] Gemini response contained no parts. Returning empty context.")
                    final_text = "[Kontxt Server: Gemini did not provide a text response.]"

                # track tokens in response
                self.token_tracker.track_response(final_text)
                
                # analyze and log response structure
                structure = analyze_response_structure(final_text)
                logger.info(f"[{request_id}] Response structure analysis: {structure['code_blocks']} code blocks, "
                           f"{structure['headings']} headings, {structure['list_items']} list items, "
                           f"{structure['paragraphs']} paragraphs, {structure['lines']} lines")
                
                if structure['heading_titles']:
                    logger.info(f"[{request_id}] Response sections: {', '.join(structure['heading_titles'])}")
                
                # log token usage summary
                usage = self.token_tracker.get_usage_summary()
                logger.info(f"[{request_id}] Token usage summary: {usage['percentage_used']:.1f}% of threshold used. "
                           f"Tool tokens: {usage['total_estimated_tokens']}, Response tokens: {usage['response_tokens']}")

                logger.info(f"[{request_id}] Gemini processing complete. Returning context (length: {len(final_text)}).")
                return final_text

            except Exception as e:
                logger.exception(f"[{request_id}] Error during Gemini interaction for query '{query}': {e}")
                error_message = f"Server error during Gemini interaction: {str(e)}"
                if hasattr(e, 'message'): # Check for common error message attribute
                     error_message = f"Server error during Gemini interaction: {e.message}"
                return error_message

    # --- internal tool implementations (called by Gemini) ---

    async def _run_subprocess(self, command: List[str], cwd: Path) -> str:
        cmd_str = ' '.join(command)
        logger.info(f"Running command: {cmd_str} in {cwd}")
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            stdout, stderr = await process.communicate()
            stdout_decoded = stdout.decode('utf-8', errors='replace').strip()
            stderr_decoded = stderr.decode('utf-8', errors='replace').strip()

            if process.returncode != 0:
                logger.error(f"Command failed (exit code {process.returncode}): {cmd_str}")
                logger.error(f"Stderr: {stderr_decoded}")
                return f"Error executing command: {cmd_str}. Exit code: {process.returncode}. Stderr: {stderr_decoded}"
            logger.info(f"Command successful: {cmd_str}")
            return stdout_decoded
        except FileNotFoundError:
            logger.error(f"Command not found: {command[0]}")
            return f"Error: Command '{command[0]}' not found. Ensure it is installed and in the PATH."
        except Exception as e:
            logger.exception(f"Failed to run subprocess {cmd_str}: {e}")
            return f"Error running subprocess: {e}"

    # create synchronous wrapper functions for the Gemini tools
    def list_repository_structure(self) -> Dict[str, Any]: # Gemini expects a dict response
        """synchronous wrapper for list_repository_structure to avoid coroutine issues"""
        tool_id = str(uuid.uuid4())[:8]  # generate a short tool call ID for tracking
        logger.info(f"[Tool:{tool_id}] Gemini calling: list_repository_structure")
        command = ["tree", "-I", "node_modules|.git|venv|.venv|__pycache__|dist|build|.next", "-a"]
        try:
            # run the command synchronously
            logger.info(f"[Tool:{tool_id}] Running tree command: {' '.join(command)}")
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                cwd=self.repo_path,
                text=True,
                check=False
            )
            if result.returncode != 0:
                logger.error(f"[Tool:{tool_id}] Command failed: {' '.join(command)}")
                return {"error": result.stderr}
            
            output = result.stdout
            # track token usage
            self.token_tracker.track_tool_usage("list_repository_structure", output)
            
            logger.info(f"[Tool:{tool_id}] Tree command successful, returned structure (bytes: {len(output)})")
            return {"structure": output}
        except Exception as e:
            logger.error(f"[Tool:{tool_id}] Error in list_repository_structure: {e}")
            return {"error": str(e)}

    def read_files(self, paths: List[str]) -> Dict[str, Any]: # Gemini expects a dict response
        """Synchronous wrapper for read_files to avoid coroutine issues"""
        tool_id = str(uuid.uuid4())[:8]  # generate a short tool call ID for tracking
        logger.info(f"[Tool:{tool_id}] Gemini calling: read_files with paths: {paths}")
        results = {}
        MAX_FILES_PER_CALL = 20
        if len(paths) > MAX_FILES_PER_CALL:
             logger.warning(f"[Tool:{tool_id}] Too many files requested ({len(paths)}), limiting to {MAX_FILES_PER_CALL}.")
             paths = paths[:MAX_FILES_PER_CALL]
             results["_warning"] = f"Too many files requested, only processed the first {MAX_FILES_PER_CALL}."

        # string to track all content for token estimation
        all_content = ""
        
        for relative_path_str in paths:
            if ".." in relative_path_str:
                 logger.warning(f"[Tool:{tool_id}] Skipping potentially unsafe path: {relative_path_str}")
                 results[relative_path_str] = "[Error: Path traversal attempted]"
                 continue

            try:
                full_path = (self.repo_path / relative_path_str).resolve()
                if self.repo_path not in full_path.parents and full_path != self.repo_path:
                     logger.warning(f"[Tool:{tool_id}] Skipping path outside repository bounds: {relative_path_str} resolved to {full_path}")
                     results[relative_path_str] = "[Error: Path outside repository bounds]"
                     continue

                if not full_path.is_file():
                     logger.warning(f"[Tool:{tool_id}] File not found or not a file: {full_path}")
                     results[relative_path_str] = "[Error: File not found or not a regular file]"
                     continue

                # read file synchronously
                logger.info(f"[Tool:{tool_id}] Reading file: {relative_path_str}")
                with open(full_path, mode='r', encoding='utf-8', errors='replace') as f:
                    MAX_FILE_SIZE_BYTES = 500 * 1024
                    content = f.read(MAX_FILE_SIZE_BYTES)
                    if len(content) == MAX_FILE_SIZE_BYTES:
                         more = f.read(1)
                         if more:
                             content += "\n... [File content truncated due to size limit]"
                             logger.warning(f"[Tool:{tool_id}] Truncated large file: {full_path}")

                    file_size = len(content)
                    logger.info(f"[Tool:{tool_id}] Successfully read: {relative_path_str} (bytes: {file_size})")
                    # log a redacted version to avoid cluttering logs
                    results[relative_path_str] = content
                    logger.info(f"[Tool:{tool_id}] File contents for {relative_path_str}: <{file_size} bytes redacted>")
                    
                    # add to all content for token tracking
                    all_content += content

            except Exception as e:
                logger.exception(f"[Tool:{tool_id}] Error reading file {relative_path_str}: {e}")
                results[relative_path_str] = f"[Error reading file: {e}]"

        # track token usage for all files combined
        self.token_tracker.track_tool_usage("read_files", all_content)
        
        logger.info(f"[Tool:{tool_id}] read_files completed, processed {len(paths)} files")
        return results

    def grep_codebase(self, pattern: str) -> Dict[str, Any]: # gemini expects a dict response
        """Synchronous wrapper for grep_codebase to avoid coroutine issues"""
        tool_id = str(uuid.uuid4())[:8]  # generate a short tool call ID for tracking
        logger.info(f"[Tool:{tool_id}] Gemini calling: grep_codebase with pattern: {pattern}")
        command = [
            "grep", "-I", "-r", "-n",
            "--exclude-dir=node_modules", "--exclude-dir=.git",
            "--exclude-dir=venv", "--exclude-dir=__pycache__",
            "-e", pattern, "."
        ]
        try:
            # run the command synchronously
            logger.info(f"[Tool:{tool_id}] Running grep command: {' '.join(command)}")
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                cwd=self.repo_path,
                text=True,
                check=False
            )
            if result.returncode != 0 and result.returncode != 1:  # grep returns 1 if no matches
                logger.error(f"[Tool:{tool_id}] Command failed: {' '.join(command)}")
                return {"error": result.stderr}
            
            output = result.stdout
            output_size = len(output)
            
            # track token usage
            self.token_tracker.track_tool_usage("grep_codebase", output)
            
            logger.info(f"[Tool:{tool_id}] Grep command successful, found results (bytes: {output_size})")
            # log a summary instead of the full results
            lines_count = output.count('\n') + 1 if output else 0
            logger.info(f"[Tool:{tool_id}] Grep results: {lines_count} matching lines, content: <{output_size} bytes redacted>")
            return {"results": output}
        except Exception as e:
            logger.error(f"[Tool:{tool_id}] Error in grep_codebase: {e}")
            return {"error": str(e)}

    # get the FastAPI app with the SSE endpoint
    def create_sse_app(self):
        """Create a FastAPI app with the SSE endpoint configured"""
        logger.info("Creating FastAPI app for SSE endpoint")
        app = FastAPI(title="Kontxt MCP Server")
        
        # configure CORS to allow connections from various clients
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("Configured CORS middleware for FastAPI app")
        
        # simple info endpoint
        @app.get("/")
        async def root():
            logger.info("Received request to root endpoint")
            return {
                "name": "Kontxt MCP Server",
                "version": "0.1.0",
                "repository_path": str(self.repo_path),
                "endpoints": {
                    "sse": "/sse"
                }
            }
            
        logger.info("Created FastAPI app with root endpoint")
        return app

# --- main execution ---
async def run_stdio_server(server_instance):
    """Run the MCP server using stdio transport"""
    logger.info("Running MCP server with stdio transport...")
    await server_instance.mcp.run_stdio()

async def run_sse_server(server_instance, host, port):
    """Run the MCP server using SSE transport"""
    sse_url = f"http://{host}:{port}/sse"
    logger.info(f"Starting SSE server at: {sse_url}")
    logger.info(f"To connect with Cursor, update your ~/.cursor/mcp.json config with:")
    logger.info(f"""
{{
  "mcpServers": {{
    "kontxt-server": {{
      "serverType": "sse",
      "url": "{sse_url}"
    }}
  }}
}}
    """)
    
    # get the SSE app from FastMCP
    app = server_instance.mcp.sse_app()
    logger.info("Retrieved SSE app from FastMCP")
    
    # run using uvicorn directly with a proper shutdown timeout
    logger.info(f"Starting Uvicorn server with host={host}, port={port}")
    config = uvicorn.Config(
        app, 
        host=host, 
        port=port,
        timeout_graceful_shutdown=3  # 3 second timeout for graceful shutdown
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    args = parse_arguments()

    if not check_tree_command():
        sys.exit(1)

    logger.info("Loading environment variables from .env file")
    load_dotenv()

    gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("Gemini API Key not found. Please provide it via --gemini-api-key or set GEMINI_API_KEY in a .env file.")
        sys.exit(1)
    else:
        logger.info("Gemini API Key found")

    repo_path = Path(args.repo_path).resolve()
    if not repo_path.is_dir():
         logger.error(f"Repository path does not exist or is not a directory: {repo_path}")
         sys.exit(1)

    logger.info(f"Initializing Kontxt MCP Server instance with repo at: {repo_path}")
    server_instance = KontxtMcpServer(
        repo_path=repo_path,
        gemini_api_key=gemini_api_key,
        token_threshold=args.token_threshold,
        gemini_model=args.gemini_model
    )

    # choose between stdio and SSE transport based on arguments
    if args.transport == "stdio":
        logger.info("Using stdio transport as specified...")
        await run_stdio_server(server_instance)
    else:
        logger.info(f"Using SSE transport with host={args.host}, port={args.port}")
        await run_sse_server(server_instance, args.host, args.port)

if __name__ == "__main__":
    try:
        logger.info("Starting Kontxt MCP Server...")
        import google.generativeai
        import mcp # check mcp import too
        logger.info("Required packages found")
    except ImportError as e:
        print(f"Error: Required package not found ({e}).")
        print("Please install the required dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    asyncio.run(main()) 