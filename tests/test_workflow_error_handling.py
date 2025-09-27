"""
Integration tests for workflow error handling and recovery mechanisms.

This module tests the comprehensive error handling system implemented in the
LangGraph workflow, including retry logic, fallback mechanisms, and monitoring.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from src.workflow.graph import (
    ConversationWorkflow,
    WorkflowError,
    ErrorType,
    ErrorSeverity,
    create_conversation_workflow
)
from src.models.conversation import ConversationState, ModelType, AgentRole
from src.workflow.flow_control import FlowDecision
from langgraph.graph import StateGraph


class TestWorkflowErrorHandling:
    """Test comprehensive error handling in the workflow."""
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample conversation state for testing."""
        return ConversationState(
            experiment_id="test-exp-001",
            persuader_model=ModelType.GPT_4O,
            persuadee_model=ModelType.LLAMA_3_3_70B,
            judge_model=ModelType.LLAMA_GUARD_2_8B,
            original_claim="Test claim for error handling",
            jailbreak_goal="Test jailbreak goal",
            max_turns=5,
            initial_agreement=2
        )
    
    @pytest.fixture
    def workflow(self):
        """Create a workflow instance for testing."""
        return create_conversation_workflow(max_retries=2)
    
    @pytest.fixture
    def mock_database_session(self):
        """Mock database session for testing."""
        with patch('src.workflow.graph.get_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            yield mock_session_instance
    
    @pytest.fixture
    def mock_agents(self):
        """Mock agent instances for testing."""
        with patch('src.agents.persuader.PersuaderAgent') as mock_persuader, \
             patch('src.agents.persuadee.PersuadeeAgent') as mock_persuadee, \
             patch('src.agents.safety_judge.SafetyJudgeAgent') as mock_judge:
            
            # Configure mock agents
            mock_persuader_instance = AsyncMock()
            mock_persuadee_instance = AsyncMock()
            mock_judge_instance = AsyncMock()
            
            mock_persuader.return_value = mock_persuader_instance
            mock_persuadee.return_value = mock_persuadee_instance
            mock_judge.return_value = mock_judge_instance
            
            yield {
                'persuader': mock_persuader_instance,
                'persuadee': mock_persuadee_instance,
                'judge': mock_judge_instance
            }


class TestWorkflowErrorClassification:
    """Test error classification and severity determination."""
    
    def test_classify_model_error(self):
        """Test classification of model-related errors."""
        workflow = ConversationWorkflow()
        
        # Test OpenAI API error
        openai_error = Exception("OpenAI API rate limit exceeded")
        error_type = workflow._classify_error(openai_error)
        assert error_type == ErrorType.MODEL_ERROR
        
        # Test model loading error
        model_error = Exception("Failed to load model weights")
        error_type = workflow._classify_error(model_error)
        assert error_type == ErrorType.MODEL_ERROR
    
    def test_classify_database_error(self):
        """Test classification of database-related errors."""
        workflow = ConversationWorkflow()
        
        # Test SQL connection error
        sql_error = Exception("database connection failed")
        error_type = workflow._classify_error(sql_error)
        assert error_type == ErrorType.DATABASE_ERROR
        
        # Test SQLAlchemy error
        class SQLAlchemyError(Exception):
            pass
        
        db_error = SQLAlchemyError("Table does not exist")
        error_type = workflow._classify_error(db_error)
        assert error_type == ErrorType.DATABASE_ERROR
    
    def test_classify_network_error(self):
        """Test classification of network-related errors."""
        workflow = ConversationWorkflow()
        
        # Test connection error
        network_error = ConnectionError("Connection refused")
        error_type = workflow._classify_error(network_error)
        assert error_type == ErrorType.NETWORK_ERROR
        
        # Test timeout error
        timeout_error = Exception("Request timeout")
        error_type = workflow._classify_error(timeout_error)
        assert error_type == ErrorType.NETWORK_ERROR
    
    def test_determine_error_severity(self):
        """Test error severity determination based on context."""
        workflow = ConversationWorkflow()
        
        # Critical node errors should be high severity
        severity = workflow._determine_error_severity(ErrorType.MODEL_ERROR, "initialize_conversation")
        assert severity == ErrorSeverity.CRITICAL
        
        # Important node errors should be high/medium severity
        severity = workflow._determine_error_severity(ErrorType.MODEL_ERROR, "persuader_turn")
        assert severity == ErrorSeverity.HIGH
        
        # Regular node errors should be medium/low severity
        severity = workflow._determine_error_severity(ErrorType.VALIDATION_ERROR, "update_conversation_state")
        assert severity == ErrorSeverity.LOW
    
    def test_is_error_recoverable(self):
        """Test error recoverability determination."""
        workflow = ConversationWorkflow()
        
        # Network errors should be recoverable
        assert workflow._is_error_recoverable(ErrorType.NETWORK_ERROR, "persuader_turn") == True
        
        # Configuration errors should not be recoverable
        assert workflow._is_error_recoverable(ErrorType.CONFIGURATION_ERROR, "persuader_turn") == False
        
        # Critical node errors should not be recoverable
        assert workflow._is_error_recoverable(ErrorType.DATABASE_ERROR, "initialize_conversation") == False


class TestWorkflowErrorHandlingIntegration:
    """Test integration of error handling in workflow nodes."""
    
    @pytest.mark.asyncio
    async def test_node_error_handling_wrapper(self, sample_state, mock_database_session):
        """Test that node error handling wrapper catches and processes errors correctly."""
        workflow = ConversationWorkflow()
        
        # Create a mock node function that raises an exception
        async def failing_node(state):
            raise ValueError("Test error in node")
        
        # Wrap the failing node with error handling
        wrapped_node = workflow._with_error_handling(failing_node, "test_node")
        
        # Execute the wrapped node
        result_state = await wrapped_node(sample_state)
        
        # Verify error was captured
        assert result_state.error_message == "Error in node 'test_node': Test error in node"
        assert hasattr(result_state, '_workflow_error')
        assert result_state._workflow_error.error_type == ErrorType.VALIDATION_ERROR
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, sample_state):
        """Test handling of timeout errors in workflow nodes."""
        workflow = ConversationWorkflow()
        
        # Create a mock node function that times out
        async def timeout_node(state):
            await asyncio.sleep(0.1)  # Simulate work
            raise asyncio.TimeoutError("Operation timed out")
        
        wrapped_node = workflow._with_error_handling(timeout_node, "timeout_test_node")
        result_state = await wrapped_node(sample_state)
        
        # Verify timeout error was handled
        assert "Timeout in node 'timeout_test_node'" in result_state.error_message
        assert hasattr(result_state, '_workflow_error')
        assert result_state._workflow_error.error_type == ErrorType.TIMEOUT_ERROR
        assert result_state._workflow_error.severity == ErrorSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, sample_state):
        """Test handling of network errors in workflow nodes."""
        workflow = ConversationWorkflow()
        
        async def network_failing_node(state):
            raise ConnectionError("Network unreachable")
        
        wrapped_node = workflow._with_error_handling(network_failing_node, "network_test_node")
        result_state = await wrapped_node(sample_state)
        
        # Verify network error was handled
        assert "Network error in node 'network_test_node'" in result_state.error_message
        assert result_state._workflow_error.error_type == ErrorType.NETWORK_ERROR
        assert result_state._workflow_error.recoverable == True


class TestWorkflowRetryMechanisms:
    """Test retry logic and fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_retry_logic_success_after_failure(self, sample_state, mock_agents):
        """Test that operations are retried and succeed after initial failure."""
        workflow = ConversationWorkflow(max_retries=2)
        
        # Configure mock agent to fail once then succeed
        mock_agents['persuader'].process_turn.side_effect = [
            Exception("Temporary failure"),
            {"content": "Successful response", "turn_number": 1}
        ]
        
        # Execute persuader turn
        result_state = await workflow._persuader_turn(sample_state)
        
        # Verify retry was attempted and succeeded
        assert mock_agents['persuader'].process_turn.call_count == 2
        assert result_state.error_message is None
        assert len(result_state.conversation_history) == 1
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, sample_state, mock_agents):
        """Test behavior when retry limit is exhausted."""
        workflow = ConversationWorkflow(max_retries=2)
        
        # Configure mock agent to always fail
        mock_agents['persuader'].process_turn.side_effect = Exception("Persistent failure")
        
        # Execute persuader turn
        result_state = await workflow._persuader_turn(sample_state)
        
        # Verify retries were exhausted
        assert mock_agents['persuader'].process_turn.call_count == 1  # Initial call only
        assert "Persuader turn failed after retries" in result_state.error_message
    
    @pytest.mark.asyncio
    async def test_retry_count_reset_on_success(self, sample_state, mock_agents):
        """Test that retry counts are reset after successful operations."""
        workflow = ConversationWorkflow(max_retries=2)
        
        # Configure mock agent to succeed
        mock_agents['persuader'].process_turn.return_value = {
            "content": "Successful response",
            "turn_number": 1
        }
        
        # Execute persuader turn
        await workflow._persuader_turn(sample_state)
        
        # Verify retry count was reset
        retry_key = f"{sample_state.experiment_id}:persuader_turn"
        assert retry_key not in workflow.flow_controller.retry_counts


class TestWorkflowErrorRecovery:
    """Test error recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_model_error_recovery(self, sample_state):
        """Test recovery from model-related errors."""
        workflow = ConversationWorkflow()
        
        # Create a model error with fallback available
        error = WorkflowError(
            message="Model API failed",
            error_type=ErrorType.MODEL_ERROR,
            context={"fallback_available": True}
        )
        
        sample_state._workflow_error = error
        
        # Attempt recovery
        recovery_success = await workflow._recover_from_model_error(sample_state, error)
        
        # Verify recovery was attempted
        assert recovery_success == True
        assert sample_state.error_message is None
        assert not hasattr(sample_state, '_workflow_error')
    
    @pytest.mark.asyncio
    async def test_database_error_recovery(self, sample_state, mock_database_session):
        """Test recovery from database-related errors."""
        workflow = ConversationWorkflow()
        
        # Configure mock database to succeed on recovery attempt
        mock_database_session.execute.return_value = None
        
        error = WorkflowError(
            message="Database connection lost",
            error_type=ErrorType.DATABASE_ERROR
        )
        
        sample_state._workflow_error = error
        
        # Attempt recovery
        recovery_success = await workflow._recover_from_database_error(sample_state, error)
        
        # Verify recovery was attempted
        assert recovery_success == True
        assert sample_state.error_message is None
    
    @pytest.mark.asyncio
    async def test_network_error_recovery(self, sample_state):
        """Test recovery from network-related errors."""
        workflow = ConversationWorkflow()
        
        error = WorkflowError(
            message="Network connection failed",
            error_type=ErrorType.NETWORK_ERROR
        )
        
        sample_state._workflow_error = error
        
        # Attempt recovery
        recovery_success = await workflow._recover_from_network_error(sample_state, error)
        
        # Verify recovery was attempted
        assert recovery_success == True
        assert sample_state.error_message is None
    
    @pytest.mark.asyncio
    async def test_non_recoverable_error_handling(self, sample_state):
        """Test handling of non-recoverable errors."""
        workflow = ConversationWorkflow()
        
        error = WorkflowError(
            message="Configuration error",
            error_type=ErrorType.CONFIGURATION_ERROR,
            recoverable=False
        )
        
        sample_state._workflow_error = error
        
        # Attempt recovery
        recovery_success = await workflow._attempt_error_recovery(sample_state)
        
        # Verify recovery was not attempted for non-recoverable error
        assert recovery_success == False


class TestWorkflowErrorLogging:
    """Test error logging and monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_error_logging_with_context(self, sample_state, caplog):
        """Test that errors are logged with comprehensive context."""
        workflow = ConversationWorkflow()
        
        # Create a workflow error with context
        error = WorkflowError(
            message="Test error for logging",
            error_type=ErrorType.MODEL_ERROR,
            severity=ErrorSeverity.HIGH,
            context={"test_context": "test_value"}
        )
        
        sample_state._workflow_error = error
        sample_state.error_message = error.message
        
        # Handle the error
        await workflow._handle_error(sample_state)
        
        # Verify error was logged with context
        assert "Test error for logging" in caplog.text
        assert sample_state.experiment_id in caplog.text
    
    @pytest.mark.asyncio
    async def test_error_metrics_logging(self, sample_state, mock_database_session):
        """Test that error metrics are properly logged."""
        workflow = ConversationWorkflow()
        
        # Set up state with some conversation history
        sample_state.current_turn = 3
        sample_state.current_agreement = 4
        sample_state.error_message = "Test error for metrics"
        
        # Log error metrics
        context = {"test": "context"}
        await workflow._log_error_metrics(sample_state, context)
        
        # Verify metrics were processed (no exceptions thrown)
        assert sample_state.error_message == "Test error for metrics"
    
    @pytest.mark.asyncio
    async def test_experiment_status_update_on_error(self, sample_state, mock_database_session):
        """Test that experiment status is updated when errors occur."""
        workflow = ConversationWorkflow()
        
        # Configure mock database
        mock_experiment = MagicMock()
        mock_database_session.query.return_value.filter_by.return_value.first.return_value = mock_experiment
        
        sample_state.error_message = "Test error for status update"
        
        # Update experiment status
        context = {}
        await workflow._update_experiment_status_failed(sample_state, context)
        
        # Verify experiment status was updated
        assert mock_experiment.status == "failed"
        assert mock_experiment.error_message == "Test error for status update"
        assert mock_experiment.completed_at is not None


class TestWorkflowTimeoutHandling:
    """Test workflow timeout handling."""
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, sample_state, mock_agents):
        """Test that workflow timeouts are handled properly."""
        workflow = ConversationWorkflow()
        
        # Configure agents to respond normally
        mock_agents['persuader'].process_turn.return_value = {
            "content": "Test response",
            "turn_number": 1
        }
        mock_agents['persuadee'].process_turn.return_value = {
            "content": "Test response",
            "agreement_score": 3,
            "turn_number": 1
        }
        mock_agents['judge'].process_turn.return_value = {
            "classification": "Safe",
            "confidence": 0.9
        }
        
        # Mock the graph to simulate timeout
        with patch.object(workflow.graph, 'ainvoke') as mock_invoke:
            mock_invoke.side_effect = asyncio.TimeoutError("Workflow timeout")
            
            # Run conversation with timeout
            result_state = await workflow.run_conversation(sample_state)
            
            # Verify timeout was handled
            assert "Workflow timeout" in result_state.error_message
            assert hasattr(result_state, '_workflow_error')
            assert result_state._workflow_error.error_type == ErrorType.TIMEOUT_ERROR


class TestWorkflowErrorIntegration:
    """Test end-to-end error handling integration."""
    
    @pytest.mark.asyncio
    async def test_complete_error_handling_flow(self, sample_state, mock_database_session, mock_agents):
        """Test complete error handling flow from error to recovery."""
        workflow = ConversationWorkflow(max_retries=1)
        
        # Configure mock experiment
        mock_experiment = MagicMock()
        mock_database_session.query.return_value.filter_by.return_value.first.return_value = mock_experiment
        
        # Configure agents to fail then succeed
        mock_agents['persuader'].process_turn.side_effect = [
            ConnectionError("Network error"),
            {"content": "Recovered response", "turn_number": 1}
        ]
        
        # Execute workflow node with error handling
        result_state = await workflow._persuader_turn(sample_state)
        
        # Verify error was handled and recovery attempted
        assert mock_agents['persuader'].process_turn.call_count == 2
        assert result_state.error_message is None
        assert len(result_state.conversation_history) == 1
    
    @pytest.mark.asyncio
    async def test_error_propagation_through_workflow(self, sample_state, mock_database_session):
        """Test that errors propagate correctly through the workflow."""
        workflow = ConversationWorkflow()
        
        # Configure mock experiment
        mock_experiment = MagicMock()
        mock_database_session.query.return_value.filter_by.return_value.first.return_value = mock_experiment
        
        # Set an error in the state
        sample_state.error_message = "Propagated error"
        
        # Handle the error
        result_state = await workflow._handle_error(sample_state)
        
        # Verify error was handled and experiment status updated
        assert result_state.error_message == "Propagated error"
        assert mock_experiment.status == "failed"
    
    @pytest.mark.asyncio
    async def test_workflow_resilience_under_multiple_errors(self, sample_state, mock_agents):
        """Test workflow resilience when multiple errors occur."""
        workflow = ConversationWorkflow(max_retries=3)
        
        # Configure different types of errors
        mock_agents['persuader'].process_turn.side_effect = [
            ConnectionError("Network error"),
            TimeoutError("Timeout error"),
            {"content": "Final success", "turn_number": 1}
        ]
        
        # Execute with multiple error types
        result_state = await workflow._persuader_turn(sample_state)
        
        # Verify workflow handled multiple errors and eventually succeeded
        assert mock_agents['persuader'].process_turn.call_count == 3
        assert result_state.error_message is None
        assert len(result_state.conversation_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])