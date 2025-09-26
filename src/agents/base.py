"""
Base agent interface and LLM integration for the Agent Jailbreak Research system.

This module provides the abstract base class for all agents and implements
LangChain model integrations with error handling and retry logic.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field

from ..models.conversation import ModelType, ConversationState
from ..config import get_settings


logger = logging.getLogger(__name__)


class RetryConfig(BaseModel):
    """Configuration for retry logic."""
    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    exponential_base: float = Field(default=2.0, ge=1.1, le=5.0)


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelTimeoutError(ModelError):
    """Exception raised when model calls timeout."""
    pass


class ModelRateLimitError(ModelError):
    """Exception raised when model rate limits are hit."""
    pass


class ModelUnavailableError(ModelError):
    """Exception raised when model is unavailable."""
    pass


class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler for logging model interactions."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Log when LLM starts processing."""
        logger.debug(f"{self.agent_name}: LLM started with {len(prompts)} prompts")
        
    def on_llm_end(self, response, **kwargs) -> None:
        """Log when LLM completes processing."""
        logger.debug(f"{self.agent_name}: LLM completed successfully")
        
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Log when LLM encounters an error."""
        logger.error(f"{self.agent_name}: LLM error - {str(error)}")


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Provides common functionality including:
    - LLM model integration with LangChain
    - Error handling and retry logic
    - Logging and monitoring
    - Configuration management
    """
    
    def __init__(
        self,
        model_type: ModelType,
        agent_name: str,
        retry_config: Optional[RetryConfig] = None,
        **model_kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            model_type: The type of model to use
            agent_name: Name of the agent for logging
            retry_config: Configuration for retry logic
            **model_kwargs: Additional arguments for model initialization
        """
        self.model_type = model_type
        self.agent_name = agent_name
        self.retry_config = retry_config or RetryConfig()
        self.settings = get_settings()
        
        # Set up logging callback before model initialization
        self.callback_handler = LoggingCallbackHandler(agent_name)
        
        # Initialize model
        self.llm = self._initialize_model(**model_kwargs)
        
        logger.info(f"Initialized {agent_name} with model {model_type.value}")
    
    def _initialize_model(self, **model_kwargs) -> Union[ChatOpenAI, HuggingFacePipeline]:
        """
        Initialize the appropriate LangChain model based on model type.
        
        Args:
            **model_kwargs: Additional model configuration
            
        Returns:
            Initialized LangChain model
            
        Raises:
            ModelError: If model initialization fails
        """
        try:
            if self.model_type == ModelType.GPT_4O:
                return self._initialize_openai_model(**model_kwargs)
            elif self.model_type in [ModelType.LLAMA_3_3_70B, ModelType.LLAMA_GUARD_2_8B]:
                return self._initialize_huggingface_model(**model_kwargs)
            else:
                raise ModelError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_type}: {str(e)}")
            raise ModelError(f"Model initialization failed: {str(e)}") from e
    
    def _initialize_openai_model(self, **model_kwargs) -> ChatOpenAI:
        """Initialize OpenAI model with default configuration."""
        default_config = {
            "model": "gpt-4o",
            "temperature": self.settings.model_temperature,
            "max_tokens": self.settings.model_max_tokens,
            "timeout": self.settings.model_timeout,
            "max_retries": 0,  # We handle retries ourselves
            "callbacks": [self.callback_handler]
        }
        default_config.update(model_kwargs)
        
        return ChatOpenAI(
            openai_api_key=self.settings.openai_api_key,
            **default_config
        )
    
    def _initialize_huggingface_model(self, **model_kwargs) -> HuggingFacePipeline:
        """Initialize HuggingFace model with default configuration."""
        # Map model types to HuggingFace model names
        model_mapping = {
            ModelType.LLAMA_3_3_70B: "meta-llama/Llama-3.3-70B-Instruct",
            ModelType.LLAMA_GUARD_2_8B: "meta-llama/LlamaGuard-2-8b"
        }
        
        model_name = model_mapping.get(self.model_type)
        if not model_name:
            raise ModelError(f"No HuggingFace mapping for {self.model_type}")
        
        default_config = {
            "model_id": model_name,
            "task": "text-generation",
            "model_kwargs": {
                "temperature": self.settings.model_temperature,
                "max_length": self.settings.model_max_tokens,
                "do_sample": True,
            },
            "callbacks": [self.callback_handler]
        }
        default_config.update(model_kwargs)
        
        return HuggingFacePipeline.from_model_id(**default_config)
    
    async def _execute_with_retry(
        self,
        operation,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with exponential backoff retry logic.
        
        Args:
            operation: The async operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            ModelError: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                logger.debug(f"{self.agent_name}: Attempt {attempt + 1}/{self.retry_config.max_retries + 1}")
                
                # Execute the operation with timeout
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.settings.model_timeout
                )
                
                if attempt > 0:
                    logger.info(f"{self.agent_name}: Succeeded after {attempt + 1} attempts")
                
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = ModelTimeoutError(f"Model call timed out after {self.settings.model_timeout}s")
                logger.warning(f"{self.agent_name}: Timeout on attempt {attempt + 1}")
                
            except Exception as e:
                # Classify the error type
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "quota" in error_msg:
                    last_exception = ModelRateLimitError(f"Rate limit exceeded: {str(e)}")
                elif "unavailable" in error_msg or "service" in error_msg:
                    last_exception = ModelUnavailableError(f"Model unavailable: {str(e)}")
                else:
                    last_exception = ModelError(f"Model error: {str(e)}")
                
                logger.warning(f"{self.agent_name}: Error on attempt {attempt + 1}: {str(e)}")
            
            # Don't wait after the last attempt
            if attempt < self.retry_config.max_retries:
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay
                )
                logger.debug(f"{self.agent_name}: Waiting {delay:.2f}s before retry")
                await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(f"{self.agent_name}: All {self.retry_config.max_retries + 1} attempts failed")
        raise last_exception
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Format messages for the specific model type.
        
        Args:
            messages: List of messages to format
            
        Returns:
            Formatted messages
        """
        # For now, return messages as-is
        # Can be overridden by subclasses for model-specific formatting
        return messages
    
    async def _call_model(self, messages: List[BaseMessage]) -> str:
        """
        Make an async call to the model with the given messages.
        
        Args:
            messages: List of messages to send to the model
            
        Returns:
            Model response as string
            
        Raises:
            ModelError: If the model call fails
        """
        formatted_messages = self._format_messages(messages)
        
        async def _make_call():
            # Convert to sync call since LangChain models are typically sync
            # In a real implementation, you might want to use aiohttp or similar
            # for truly async calls
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.invoke(formatted_messages)
            )
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        
        return await self._execute_with_retry(_make_call)
    
    @abstractmethod
    async def process_turn(self, state: ConversationState) -> Dict[str, Any]:
        """
        Process a conversation turn and return the result.
        
        This method must be implemented by all concrete agent classes.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dictionary containing the agent's response and any additional data
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Returns:
            System prompt string
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "agent_name": self.agent_name,
            "model_type": self.model_type.value,
            "temperature": self.settings.model_temperature,
            "max_tokens": self.settings.model_max_tokens,
            "timeout": self.settings.model_timeout,
            "retry_config": self.retry_config.model_dump()
        }
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the model.
        
        Returns:
            True if the model is healthy, False otherwise
        """
        try:
            test_messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Say 'OK' if you can hear me.")
            ]
            
            response = await self._call_model(test_messages)
            return "ok" in response.lower()
            
        except Exception as e:
            logger.error(f"{self.agent_name}: Health check failed: {str(e)}")
            return False