import unittest
import asyncio
import os
import sys
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import kontxt_server
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcp import ClientSession
from kontxt_server import KontxtMcpServer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

class TestKontxtMcpServer(unittest.TestCase):
    """Test suite for the KontxtMcpServer functionality."""

    def setUp(self):
        """Set up the test environment before each test."""
        # Mock API key for testing
        self.api_key = "fake_api_key"
        # Use the current repo as the test repo
        self.repo_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        self.token_threshold = 800000
        self.model_name = "models/gemini-2.5-flash-preview-04-17"

    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_server_initialization(self, mock_configure, mock_generative_model):
        """Test that the server initializes properly."""
        # Create a mock instance for the GenerativeModel
        mock_generative_model.return_value = MagicMock()
        
        # Initialize the server
        server = KontxtMcpServer(
            repo_path=self.repo_path,
            gemini_api_key=self.api_key,
            token_threshold=self.token_threshold,
            gemini_model=self.model_name
        )
        
        # Assert that the API was configured with our key
        mock_configure.assert_called_once_with(api_key=self.api_key)
        
        # Assert that the GenerativeModel was initialized with the expected model name
        mock_generative_model.assert_called_once()
        self.assertEqual(server.gemini_model_name, self.model_name)
        
        # Verify server attributes are set correctly
        self.assertEqual(server.repo_path, self.repo_path)
        self.assertEqual(server.token_threshold, self.token_threshold)
        logger.info("Server initialization test passed")

    async def test_gemini_client_integration(self):
        """Test that Gemini client is properly set up in the server."""
        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_generative_model:
            
            # Mock the GenerativeModel
            mock_model = MagicMock()
            mock_generative_model.return_value = mock_model
            
            # Initialize the server with mocks
            server = KontxtMcpServer(
                repo_path=self.repo_path,
                gemini_api_key=self.api_key,
                token_threshold=self.token_threshold,
                gemini_model=self.model_name
            )
            
            # Verify API was configured with our key
            mock_configure.assert_called_once_with(api_key=self.api_key)
            
            # Verify GenerativeModel was initialized with the expected model name
            mock_generative_model.assert_called_once()
            self.assertEqual(server.gemini_model_name, self.model_name)
            
            # Verify server has a gemini_client attribute
            self.assertIsNotNone(server.gemini_client)
            
            # Verify the tools configuration was set up
            self.assertIsNotNone(server.gemini_tools_config)
            self.assertEqual(len(server.gemini_tool_schemas), 3)
            
            logger.info("Gemini client integration test passed")

    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_list_repository_structure_tool(self, mock_configure, mock_generative_model):
        """Test the list_repository_structure tool."""
        # Mock the GenerativeModel
        mock_generative_model.return_value = MagicMock()
        
        # Create a mock subprocess.run result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "mock directory structure"
        
        # Initialize the server
        server = KontxtMcpServer(
            repo_path=self.repo_path,
            gemini_api_key=self.api_key,
            token_threshold=self.token_threshold,
            gemini_model=self.model_name
        )
        
        # Path the subprocess.run to return our mock result
        with patch('subprocess.run', return_value=mock_result):
            # Call the tool method
            result = server.list_repository_structure()
            
            # Verify the result contains the expected structure
            self.assertEqual(result["structure"], "mock directory structure")
        
        logger.info("list_repository_structure test passed")

def main():
    # Create test suite for sync tests
    sync_test_suite = unittest.TestSuite()
    sync_test_suite.addTest(TestKontxtMcpServer('test_server_initialization'))
    sync_test_suite.addTest(TestKontxtMcpServer('test_list_repository_structure_tool'))
    
    # Run the synchronous tests
    sync_runner = unittest.TextTestRunner()
    sync_result = sync_runner.run(sync_test_suite)
    
    # Check if sync tests passed
    if not sync_result.wasSuccessful():
        return
    
    # Run the async test separately
    async_test = TestKontxtMcpServer('test_gemini_client_integration')
    async_test.setUp()
    
    # Set up and run the event loop for async tests
    async def run_async_test():
        try:
            await async_test.test_gemini_client_integration()
            logger.info("Async test passed")
            return True
        except Exception as e:
            logger.error(f"Async test failed: {e}")
            return False
    
    # Use new asyncio API to avoid deprecation warnings
    async_result = asyncio.run(run_async_test())
    
    if async_result:
        logger.info("All tests completed successfully")
    else:
        logger.error("Async tests failed")

if __name__ == "__main__":
    main()