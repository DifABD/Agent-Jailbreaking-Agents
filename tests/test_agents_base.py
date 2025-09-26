"""
Unit tests for the base agent interface and LLM integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.agents.base import (
    BaseAgent,
    RetryConfig,
    ModelError,
    ModelTimeoutError,
    ModelRateLimitError,
    ModelUnavailableError,
    LoggingCallbackHandler
)
from src.models.conversation import ModelType, ConversationState
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing."""
    
    async def process_turn(self, state: ConversationState):
        return {"response": "test response", "data": "test data"}
    
    def get_system_prompt(self):
        return "You are a test agent."


class TestRetryConfig:
    """Test RetryConfig model."""
    
    def test_default_values(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
    
    def test_custom_values(self):
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
    
    def test_validation(self):
        # Test invalid values
        with pytest.raises(ValueError):
            RetryConfig(max_retries=-1)
        
        with pytest.raises(ValueError):
            RetryConfig(base_delay=0.0)
        
        with pytest.raises(ValueError):
            RetryConfig(exponential_base=1.0)


class TestLoggingCallbackHandler:
    """Test LoggingCallbackHandler."""
    
    def test_initialization(self):
        handler = LoggingCallbackHandler("test_agent")
        assert handler.agent_name == "test_agent"
    
    @patch('src.agents.base.logger')
    def test_on_llm_start(self, mock_logger):
        handler = LoggingCallbackHandler("test_agent")
        handler.on_llm_start({}, ["prompt1", "prompt2"])
        mock_logger.debug.assert_called_once_with("test_agent: LLM started with 2 prompts")
    
    @patch('src.agents.base.logger')
    def test_on_llm_end(self, mock_logger):
        handler = LoggingCallbackHandler("test_agent")
        handler.on_llm_end("response")
        mock_logger.debug.assert_called_once_with("test_agent: LLM completed successfully")
    
    @patch('src.agents.base.logger')
    def test_on_llm_error(self, mock_logger):
        handler = LoggingCallbackHandler("test_agent")
        error = Exception("test error")
        handler.on_llm_error(error)
        mock_logger.error.assert_called_once_with("test_agent: LLM error - test error")


class TestBaseAgent:
    """Test BaseAgent functionality."""
    
    @patch('src.agents.base.get_settings')
    def test_initialization_gpt4o(self, mock_get_settings):
        # Mock settings
        mock_settings = Mock()
        mock_settings.model_temperature = 0.7
        mock_settings.model_max_tokens = 1000
        mock_settings.model_timeout = 30
        mock_settings.openai_api_key = "test-key"
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            
            assert agent.model_type == ModelType.GPT_4O
            assert agent.agent_name == "test_agent"
            assert isinstance(agent.retry_config, RetryConfig)
            mock_openai.assert_called_once()
    
    @patch('src.agents.base.get_settings')
    def test_initialization_llama(self, mock_get_settings):
        # Mock settings
        mock_settings = Mock()
        mock_settings.model_temperature = 0.7
        mock_settings.model_max_tokens = 1000
        mock_settings.model_timeout = 30
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.HuggingFacePipeline') as mock_hf:
            mock_hf.from_model_id.return_value = Mock()
            
            agent = TestAgent(ModelType.LLAMA_3_3_70B, "test_agent")
            
            assert agent.model_type == ModelType.LLAMA_3_3_70B
            mock_hf.from_model_id.assert_called_once()
    
    @patch('src.agents.base.get_settings')
    def test_initialization_unsupported_model(self, mock_get_settings):
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        # Create a mock ModelType that's not supported
        with pytest.raises(ModelError, match="Unsupported model type"):
            TestAgent("unsupported_model", "test_agent")
    
    @patch('src.agents.base.get_settings')
    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self, mock_get_settings):
        mock_settings = Mock()
        mock_settings.model_timeout = 30
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI'):
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            
            # Mock successful operation
            async def mock_operation():
                return "success"
            
            result = await agent._execute_with_retry(mock_operation)
            assert result == "success"
    
    @patch('src.agents.base.get_settings')
    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_failure(self, mock_get_settings):
        mock_settings = Mock()
        mock_settings.model_timeout = 30
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI'):
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            agent.retry_config = RetryConfig(max_retries=2, base_delay=0.1)
            
            call_count = 0
            
            async def mock_operation():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("temporary error")
                return "success"
            
            with patch('asyncio.sleep'):  # Speed up test
                result = await agent._execute_with_retry(mock_operation)
                assert result == "success"
                assert call_count == 2
    
    @patch('src.agents.base.get_settings')
    @pytest.mark.asyncio
    async def test_execute_with_retry_timeout_error(self, mock_get_settings):
        mock_settings = Mock()
        mock_settings.model_timeout = 0.1  # Very short timeout
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI'):
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            agent.retry_config = RetryConfig(max_retries=0, base_delay=0.1)  # No retries to speed up test
            
            async def mock_operation():
                # Simulate a timeout by raising TimeoutError directly
                raise asyncio.TimeoutError("Operation timed out")
            
            with pytest.raises(ModelTimeoutError):
                await agent._execute_with_retry(mock_operation)
    
    @patch('src.agents.base.get_settings')
    @pytest.mark.asyncio
    async def test_execute_with_retry_rate_limit_error(self, mock_get_settings):
        mock_settings = Mock()
        mock_settings.model_timeout = 30
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI'):
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            agent.retry_config = RetryConfig(max_retries=1, base_delay=0.1)
            
            async def mock_operation():
                raise Exception("rate limit exceeded")
            
            with patch('asyncio.sleep'):  # Speed up test
                with pytest.raises(ModelRateLimitError):
                    await agent._execute_with_retry(mock_operation)
    
    @patch('src.agents.base.get_settings')
    @pytest.mark.asyncio
    async def test_call_model(self, mock_get_settings):
        mock_settings = Mock()
        mock_settings.model_timeout = 30
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "test response"
            mock_llm.invoke.return_value = mock_response
            mock_openai.return_value = mock_llm
            
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            
            messages = [HumanMessage(content="test")]
            
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_executor = AsyncMock()
                mock_executor.return_value = mock_response
                mock_loop.return_value.run_in_executor = mock_executor
                
                result = await agent._call_model(messages)
                assert result == "test response"
    
    @patch('src.agents.base.get_settings')
    def test_get_model_info(self, mock_get_settings):
        mock_settings = Mock()
        mock_settings.model_temperature = 0.7
        mock_settings.model_max_tokens = 1000
        mock_settings.model_timeout = 30
        mock_settings.openai_api_key = "test-key"
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI'):
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            
            info = agent.get_model_info()
            
            assert info["agent_name"] == "test_agent"
            assert info["model_type"] == "gpt-4o"
            assert info["temperature"] == 0.7
            assert info["max_tokens"] == 1000
            assert info["timeout"] == 30
            assert "retry_config" in info
    
    @patch('src.agents.base.get_settings')
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_get_settings):
        mock_settings = Mock()
        mock_settings.model_timeout = 30
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI'):
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            
            # Mock successful health check
            with patch.object(agent, '_call_model', return_value="OK"):
                result = await agent.health_check()
                assert result is True
    
    @patch('src.agents.base.get_settings')
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_get_settings):
        mock_settings = Mock()
        mock_settings.model_timeout = 30
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI'):
            agent = TestAgent(ModelType.GPT_4O, "test_agent")
            
            # Mock failed health check
            with patch.object(agent, '_call_model', side_effect=Exception("Model error")):
                result = await agent.health_check()
                assert result is False
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            BaseAgent(ModelType.GPT_4O, "test")