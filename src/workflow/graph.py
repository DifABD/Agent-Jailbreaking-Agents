"""
LangGraph workflow implementation for agent jailbreaking research.

This module contains the core LangGraph workflow that orchestrates the conversation
between persuader and persuadee agents, manages state transitions, and handles
the complete experimental pipeline.
"""

import logging
import traceback
import asyncio
import json
from typing import Dict, Any, Optional, Literal, Union
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from ..models.conversation import ConversationState, AgentRole, ExperimentStatus, ModelType
    from ..database.connection import get_session, get_global_session_factory
    from ..models.database import Experiment, ConversationTurn
    from .flow_control import ConversationFlowController, TurnManager, FlowDecision
except ImportError:
    # Fallback for direct execution
    from src.models.conversation import ConversationState, AgentRole, ExperimentStatus, ModelType
    from src.database.connection import get_session, get_global_session_factory
    from src.models.database import Experiment, ConversationTurn
    from src.workflow.flow_control import ConversationFlowController, TurnManager, FlowDecision


logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Types of errors that can occur in the workflow."""
    AGENT_ERROR = "agent_error"
    DATABASE_ERROR = "database_error"
    MODEL_ERROR = "model_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkflowError(Exception):
    """Custom exception for workflow errors with enhanced metadata."""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.recoverable = recoverable
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and storage."""
        return {
            "message": str(self),
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "original_error": str(self.original_error) if self.original_error else None,
            "traceback": traceback.format_exc() if self.original_error else None
        }


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for preventing cascade failures.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if operation can be executed based on circuit state."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).seconds >= self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class WorkflowMonitor:
    """
    Monitors workflow execution and provides metrics and alerts.
    """
    
    def __init__(self):
        """Initialize workflow monitor."""
        self.metrics = {
            "total_experiments": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "error_counts": {},
            "average_duration": 0.0,
            "circuit_breaker_trips": 0
        }
        self.active_experiments = {}
    
    def start_experiment(self, experiment_id: str, start_time: datetime):
        """Record experiment start."""
        self.active_experiments[experiment_id] = {
            "start_time": start_time,
            "errors": []
        }
        self.metrics["total_experiments"] += 1
    
    def record_error(self, experiment_id: str, error: WorkflowError):
        """Record error for monitoring."""
        if experiment_id in self.active_experiments:
            self.active_experiments[experiment_id]["errors"].append(error)
        
        error_key = f"{error.error_type.value}:{error.severity.value}"
        self.metrics["error_counts"][error_key] = self.metrics["error_counts"].get(error_key, 0) + 1
    
    def complete_experiment(self, experiment_id: str, success: bool, end_time: datetime):
        """Record experiment completion."""
        if experiment_id in self.active_experiments:
            start_time = self.active_experiments[experiment_id]["start_time"]
            duration = (end_time - start_time).total_seconds()
            
            # Update average duration
            total_completed = self.metrics["successful_experiments"] + self.metrics["failed_experiments"]
            if total_completed > 0:
                self.metrics["average_duration"] = (
                    (self.metrics["average_duration"] * total_completed + duration) / (total_completed + 1)
                )
            else:
                self.metrics["average_duration"] = duration
            
            if success:
                self.metrics["successful_experiments"] += 1
            else:
                self.metrics["failed_experiments"] += 1
            
            del self.active_experiments[experiment_id]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the workflow system."""
        total_completed = self.metrics["successful_experiments"] + self.metrics["failed_experiments"]
        success_rate = (
            self.metrics["successful_experiments"] / total_completed 
            if total_completed > 0 else 0.0
        )
        
        return {
            "success_rate": success_rate,
            "active_experiments": len(self.active_experiments),
            "total_experiments": self.metrics["total_experiments"],
            "error_counts": self.metrics["error_counts"],
            "average_duration": self.metrics["average_duration"],
            "health_status": "healthy" if success_rate > 0.8 else "degraded" if success_rate > 0.5 else "unhealthy"
        }


class ConversationWorkflow:
    """
    Main LangGraph workflow for managing agent conversations.
    
    This class encapsulates the complete conversation workflow including
    initialization, turn-based interactions, safety evaluation, and
    result calculation.
    """
    
    def __init__(self, checkpointer_path: Optional[str] = None, max_retries: int = 3):
        """
        Initialize the conversation workflow.
        
        Args:
            checkpointer_path: Path to SQLite database for state checkpointing.
                              If None, uses in-memory database.
            max_retries: Maximum number of retries for failed operations.
        """
        self.checkpointer_path = checkpointer_path or ":memory:"
        # Use MemorySaver for now - can be extended to SQLite later
        self.checkpointer = MemorySaver()
        self.flow_controller = ConversationFlowController(max_retries=max_retries)
        self.turn_manager = TurnManager()
        self.circuit_breakers = {
            "model_api": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            "database": CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        }
        self.monitor = WorkflowMonitor()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with all nodes and edges.
        
        Returns:
            StateGraph: Compiled workflow graph
        """
        # Create the state graph
        workflow = StateGraph(ConversationState)
        
        # Add workflow nodes with error handling wrappers
        workflow.add_node("initialize_conversation", self._with_error_handling(self._initialize_conversation, "initialize_conversation"))
        workflow.add_node("load_claim", self._with_error_handling(self._load_claim, "load_claim"))
        workflow.add_node("set_initial_stance", self._with_error_handling(self._set_initial_stance, "set_initial_stance"))
        workflow.add_node("persuader_turn", self._with_error_handling(self._persuader_turn, "persuader_turn"))
        workflow.add_node("update_conversation_state", self._with_error_handling(self._update_conversation_state, "update_conversation_state"))
        workflow.add_node("persuadee_evaluation", self._with_error_handling(self._persuadee_evaluation, "persuadee_evaluation"))
        workflow.add_node("check_turn_limit", self._with_error_handling(self._check_turn_limit, "check_turn_limit"))
        workflow.add_node("generate_final_response", self._with_error_handling(self._generate_final_response, "generate_final_response"))
        workflow.add_node("safety_judge", self._with_error_handling(self._safety_judge, "safety_judge"))
        workflow.add_node("calculate_metrics", self._with_error_handling(self._calculate_metrics, "calculate_metrics"))
        workflow.add_node("analyze_strategies", self._with_error_handling(self._analyze_strategies, "analyze_strategies"))
        workflow.add_node("finalize_experiment", self._with_error_handling(self._finalize_experiment, "finalize_experiment"))
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("initialize_conversation")
        
        # Add edges for the main flow
        workflow.add_edge("initialize_conversation", "load_claim")
        workflow.add_edge("load_claim", "set_initial_stance")
        workflow.add_edge("set_initial_stance", "persuader_turn")
        workflow.add_edge("persuader_turn", "update_conversation_state")
        workflow.add_edge("update_conversation_state", "persuadee_evaluation")
        workflow.add_edge("persuadee_evaluation", "check_turn_limit")
        
        # Add conditional routing from turn limit check
        workflow.add_conditional_edges(
            "check_turn_limit",
            self._should_continue_conversation,
            {
                "continue_conversation": "persuader_turn",
                "end_conversation": "generate_final_response",
                "handle_error": "handle_error"
            }
        )
        
        # Add final processing flow
        workflow.add_edge("generate_final_response", "safety_judge")
        workflow.add_edge("safety_judge", "calculate_metrics")
        workflow.add_edge("calculate_metrics", "analyze_strategies")
        workflow.add_edge("analyze_strategies", "finalize_experiment")
        
        # Add termination edges
        workflow.add_edge("finalize_experiment", END)
        workflow.add_edge("handle_error", END)
        
        # Compile the graph with checkpointing
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _with_error_handling(self, node_func, node_name: str):
        """
        Wrap a workflow node with comprehensive error handling.
        
        Args:
            node_func: The original node function
            node_name: Name of the node for logging
            
        Returns:
            Wrapped function with error handling
        """
        async def wrapped_node(state: ConversationState) -> ConversationState:
            start_time = datetime.utcnow()
            
            # Check circuit breaker before execution
            circuit_breaker_key = self._get_circuit_breaker_key(node_name)
            if circuit_breaker_key and not self.circuit_breakers[circuit_breaker_key].can_execute():
                error = WorkflowError(
                    message=f"Circuit breaker open for node '{node_name}'",
                    error_type=ErrorType.CONFIGURATION_ERROR,
                    severity=ErrorSeverity.HIGH,
                    recoverable=False,
                    context={
                        "node_name": node_name,
                        "circuit_breaker": circuit_breaker_key
                    }
                )
                return self._handle_node_error(state, error, node_name)
            
            try:
                logger.debug(f"Starting node '{node_name}' for experiment {state.experiment_id}")
                
                # Execute the original node function with timeout
                node_timeout = self._get_node_timeout(node_name)
                result_state = await asyncio.wait_for(node_func(state), timeout=node_timeout)
                
                # Record success in circuit breaker
                if circuit_breaker_key:
                    self.circuit_breakers[circuit_breaker_key].record_success()
                
                # Log successful completion
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.debug(
                    f"Node '{node_name}' completed successfully for experiment {state.experiment_id}",
                    extra={
                        "node_name": node_name,
                        "duration_seconds": duration,
                        "experiment_id": state.experiment_id
                    }
                )
                
                return result_state
                
            except asyncio.TimeoutError as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                error = WorkflowError(
                    message=f"Timeout in node '{node_name}': {str(e)}",
                    error_type=ErrorType.TIMEOUT_ERROR,
                    severity=ErrorSeverity.HIGH,
                    recoverable=True,
                    context={
                        "node_name": node_name,
                        "duration_seconds": duration,
                        "timeout_threshold": node_timeout
                    },
                    original_error=e
                )
                
                # Record failure in circuit breaker
                if circuit_breaker_key:
                    self.circuit_breakers[circuit_breaker_key].record_failure()
                
                return self._handle_node_error(state, error, node_name)
                
            except ConnectionError as e:
                error = WorkflowError(
                    message=f"Network error in node '{node_name}': {str(e)}",
                    error_type=ErrorType.NETWORK_ERROR,
                    severity=ErrorSeverity.MEDIUM,
                    recoverable=True,
                    context={"node_name": node_name},
                    original_error=e
                )
                
                # Record failure in circuit breaker
                if circuit_breaker_key:
                    self.circuit_breakers[circuit_breaker_key].record_failure()
                
                return self._handle_node_error(state, error, node_name)
                
            except Exception as e:
                # Classify error type based on exception
                error_type = self._classify_error(e)
                severity = self._determine_error_severity(error_type, node_name)
                recoverable = self._is_error_recoverable(error_type, node_name)
                
                error = WorkflowError(
                    message=f"Error in node '{node_name}': {str(e)}",
                    error_type=error_type,
                    severity=severity,
                    recoverable=recoverable,
                    context={
                        "node_name": node_name,
                        "exception_type": type(e).__name__
                    },
                    original_error=e
                )
                
                # Record failure in circuit breaker for certain error types
                if circuit_breaker_key and error_type in [ErrorType.MODEL_ERROR, ErrorType.DATABASE_ERROR]:
                    self.circuit_breakers[circuit_breaker_key].record_failure()
                
                return self._handle_node_error(state, error, node_name)
        
        return wrapped_node
    
    def _get_circuit_breaker_key(self, node_name: str) -> Optional[str]:
        """
        Get the appropriate circuit breaker key for a node.
        
        Args:
            node_name: Name of the workflow node
            
        Returns:
            Optional[str]: Circuit breaker key or None if no circuit breaker needed
        """
        if node_name in ["persuader_turn", "persuadee_evaluation", "safety_judge"]:
            return "model_api"
        elif node_name in ["initialize_conversation", "update_conversation_state", "finalize_experiment"]:
            return "database"
        return None
    
    def _get_node_timeout(self, node_name: str) -> float:
        """
        Get timeout value for a specific node.
        
        Args:
            node_name: Name of the workflow node
            
        Returns:
            float: Timeout in seconds
        """
        # Different timeouts for different node types
        timeout_map = {
            "persuader_turn": 60.0,
            "persuadee_evaluation": 60.0,
            "safety_judge": 45.0,
            "generate_final_response": 60.0,
            "initialize_conversation": 30.0,
            "load_claim": 10.0,
            "set_initial_stance": 10.0,
            "update_conversation_state": 15.0,
            "check_turn_limit": 5.0,
            "calculate_metrics": 10.0,
            "analyze_strategies": 30.0,
            "finalize_experiment": 20.0
        }
        
        return timeout_map.get(node_name, 30.0)  # Default 30 seconds
    
    def _handle_node_error(self, state: ConversationState, error: WorkflowError, node_name: str) -> ConversationState:
        """
        Handle errors that occur in workflow nodes.
        
        Args:
            state: Current conversation state
            error: The workflow error that occurred
            node_name: Name of the node where error occurred
            
        Returns:
            ConversationState: Updated state with error information
        """
        # Store error in state for the error handler
        state._workflow_error = error
        state.error_message = error.message
        
        # Record error in monitoring system
        self.monitor.record_error(state.experiment_id, error)
        
        # Log the error with full context
        logger.error(
            f"Error in node '{node_name}' for experiment {state.experiment_id}: {error.message}",
            extra={
                **error.to_dict(),
                "workflow_health": self.monitor.get_health_status()
            }
        )
        
        # Check circuit breaker status and log if tripped
        circuit_breaker_key = self._get_circuit_breaker_key(node_name)
        if circuit_breaker_key and self.circuit_breakers[circuit_breaker_key].state == "open":
            logger.warning(
                f"Circuit breaker '{circuit_breaker_key}' is open due to repeated failures",
                extra={
                    "circuit_breaker": circuit_breaker_key,
                    "failure_count": self.circuit_breakers[circuit_breaker_key].failure_count,
                    "experiment_id": state.experiment_id
                }
            )
            self.monitor.metrics["circuit_breaker_trips"] += 1
        
        # Check if we should attempt immediate retry
        if error.recoverable and self.flow_controller.should_retry_operation(state, node_name):
            logger.info(f"Attempting immediate retry for node '{node_name}'")
            # Clear error for retry
            state.error_message = None
            delattr(state, '_workflow_error')
        
        return state
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        """
        Classify an exception into an error type.
        
        Args:
            exception: The exception to classify
            
        Returns:
            ErrorType: The classified error type
        """
        exception_name = type(exception).__name__.lower()
        error_message = str(exception).lower()
        
        # Model-related errors
        if any(keyword in exception_name for keyword in ['model', 'llm', 'openai', 'huggingface']):
            return ErrorType.MODEL_ERROR
        if any(keyword in error_message for keyword in ['model', 'api key', 'rate limit', 'quota']):
            return ErrorType.MODEL_ERROR
            
        # Database errors
        if any(keyword in exception_name for keyword in ['sql', 'database', 'connection']):
            return ErrorType.DATABASE_ERROR
        if any(keyword in error_message for keyword in ['database', 'connection', 'sql']):
            return ErrorType.DATABASE_ERROR
            
        # Network errors
        if any(keyword in exception_name for keyword in ['connection', 'network', 'timeout', 'http']):
            return ErrorType.NETWORK_ERROR
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'unreachable']):
            return ErrorType.NETWORK_ERROR
            
        # Validation errors
        if any(keyword in exception_name for keyword in ['validation', 'value', 'type']):
            return ErrorType.VALIDATION_ERROR
        if any(keyword in error_message for keyword in ['invalid', 'validation', 'required']):
            return ErrorType.VALIDATION_ERROR
            
        # Timeout errors
        if 'timeout' in exception_name:
            return ErrorType.TIMEOUT_ERROR
            
        return ErrorType.UNKNOWN_ERROR
    
    def _determine_error_severity(self, error_type: ErrorType, node_name: str) -> ErrorSeverity:
        """
        Determine the severity of an error based on type and context.
        
        Args:
            error_type: The type of error
            node_name: The node where the error occurred
            
        Returns:
            ErrorSeverity: The determined severity level
        """
        # Critical nodes where errors are more severe
        critical_nodes = ["initialize_conversation", "finalize_experiment"]
        important_nodes = ["persuader_turn", "persuadee_evaluation", "safety_judge"]
        
        if node_name in critical_nodes:
            return ErrorSeverity.CRITICAL
        elif node_name in important_nodes:
            if error_type in [ErrorType.MODEL_ERROR, ErrorType.DATABASE_ERROR]:
                return ErrorSeverity.HIGH
            else:
                return ErrorSeverity.MEDIUM
        else:
            if error_type == ErrorType.DATABASE_ERROR:
                return ErrorSeverity.HIGH
            elif error_type in [ErrorType.MODEL_ERROR, ErrorType.NETWORK_ERROR]:
                return ErrorSeverity.MEDIUM
            else:
                return ErrorSeverity.LOW
    
    def _is_error_recoverable(self, error_type: ErrorType, node_name: str) -> bool:
        """
        Determine if an error is recoverable.
        
        Args:
            error_type: The type of error
            node_name: The node where the error occurred
            
        Returns:
            bool: True if the error is potentially recoverable
        """
        # Generally recoverable error types
        recoverable_types = [
            ErrorType.NETWORK_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.MODEL_ERROR  # May have fallback models
        ]
        
        # Configuration errors are usually not recoverable
        non_recoverable_types = [
            ErrorType.CONFIGURATION_ERROR
        ]
        
        if error_type in non_recoverable_types:
            return False
        elif error_type in recoverable_types:
            return True
        else:
            # For other types, depend on the node
            critical_nodes = ["initialize_conversation"]
            return node_name not in critical_nodes
    
    async def _initialize_conversation(self, state: ConversationState) -> ConversationState:
        """
        Initialize the conversation workflow.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state with initialization
        """
        logger.info(f"Initializing conversation for experiment {state.experiment_id}")
        
        try:
            # Update database with experiment start
            with get_session() as session:
                experiment = session.query(Experiment).filter_by(id=state.experiment_id).first()
                if experiment:
                    experiment.status = "running"
                    experiment.started_at = datetime.utcnow()
                    session.commit()
                    logger.info(f"Updated experiment {state.experiment_id} status to running")
                else:
                    logger.warning(f"Experiment {state.experiment_id} not found in database")
            
            # Reset conversation state for fresh start
            state.conversation_history = []
            state.current_turn = 0
            state.agreement_scores = []
            state.current_agreement = state.initial_agreement
            state.is_completed = False
            state.error_message = None
            
            logger.info(f"Conversation initialized successfully for experiment {state.experiment_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error initializing conversation: {str(e)}")
            state.error_message = f"Initialization failed: {str(e)}"
            return state
    
    async def _load_claim(self, state: ConversationState) -> ConversationState:
        """
        Load and validate the claim for the conversation with data loading integration.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state with loaded claim
        """
        logger.info(f"Loading claim for experiment {state.experiment_id}")
        
        try:
            # Validate that claim is present and non-empty
            if not state.original_claim or not state.original_claim.strip():
                raise ValueError("Original claim is empty or missing")
            
            if not state.jailbreak_goal or not state.jailbreak_goal.strip():
                raise ValueError("Jailbreak goal is empty or missing")
            
            # If claim loading from external data is needed, integrate with data loader
            # For now, we assume the claim is already loaded in the state
            
            # Validate claim format and content
            if len(state.original_claim) < 10:
                raise ValueError("Claim is too short to be meaningful")
            
            if len(state.jailbreak_goal) < 10:
                raise ValueError("Jailbreak goal is too short to be meaningful")
            
            # Log claim details for debugging
            logger.info(f"Claim loaded successfully: {state.original_claim[:100]}...")
            logger.info(f"Jailbreak goal: {state.jailbreak_goal[:100]}...")
            
            return state
            
        except Exception as e:
            logger.error(f"Error loading claim: {str(e)}")
            state.error_message = f"Claim loading failed: {str(e)}"
            return state
    
    async def _set_initial_stance(self, state: ConversationState) -> ConversationState:
        """
        Set the initial stance for the persuadee agent.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state with initial stance
        """
        logger.info(f"Setting initial stance for experiment {state.experiment_id}")
        
        try:
            # Initialize agreement tracking
            state.current_agreement = state.initial_agreement
            state.agreement_scores = [state.initial_agreement]
            
            # Add initial stance as a system message
            initial_message = f"Initial stance on claim '{state.original_claim}': Agreement level {state.initial_agreement}/5"
            
            # This is not added as a conversation turn, but logged for reference
            logger.info(f"Initial stance set: {initial_message}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error setting initial stance: {str(e)}")
            state.error_message = f"Initial stance setting failed: {str(e)}"
            return state
    
    async def _persuader_turn(self, state: ConversationState) -> ConversationState:
        """
        Handle persuader agent turn with actual agent integration.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state after persuader turn
        """
        logger.info(f"Processing persuader turn {state.current_turn} for experiment {state.experiment_id}")
        
        try:
            # Validate turn sequence
            if not self.flow_controller.validate_turn_sequence(state):
                raise ValueError("Invalid turn sequence detected")
            
            # Verify it's the persuader's turn
            expected_speaker = self.flow_controller.get_next_speaker(state)
            if expected_speaker != AgentRole.PERSUADER:
                raise ValueError(f"Expected {expected_speaker}, but processing persuader turn")
            
            # Initialize persuader agent if not already done
            if not hasattr(self, '_persuader_agent') or self._persuader_agent is None:
                try:
                    from ..agents.persuader import PersuaderAgent
                except ImportError:
                    from src.agents.persuader import PersuaderAgent
                self._persuader_agent = PersuaderAgent(
                    model_type=state.persuader_model,
                    agent_name=f"Persuader-{state.experiment_id}"
                )
            
            # Process turn with persuader agent
            turn_result = await self._persuader_agent.process_turn(state)
            
            # Validate turn content
            if not self.turn_manager.validate_turn_content(turn_result["content"], AgentRole.PERSUADER):
                raise ValueError("Invalid turn content generated")
            
            # Add turn to conversation history
            state.add_turn(
                speaker=AgentRole.PERSUADER,
                content=turn_result["content"]
            )
            
            # Store strategy hints if available
            if "strategy_hints" in turn_result:
                strategy_data = {
                    "turn_number": turn_result["turn_number"],
                    "strategy_category": "persuasion",
                    "strategy_hints": turn_result["strategy_hints"],
                    "confidence": 0.7  # Default confidence for heuristic detection
                }
                state.strategies_used.append(strategy_data)
            
            # Reset retry count for successful operation
            self.flow_controller.reset_retry_count(state, "persuader_turn")
            
            logger.info(f"Persuader turn {state.current_turn - 1} completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in persuader turn: {str(e)}")
            
            # Check if we should retry
            if self.flow_controller.should_retry_operation(state, "persuader_turn"):
                logger.info("Retrying persuader turn")
                return await self._persuader_turn(state)
            else:
                state.error_message = f"Persuader turn failed after retries: {str(e)}"
                return state
    
    async def _update_conversation_state(self, state: ConversationState) -> ConversationState:
        """
        Update conversation state after each turn.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state with persisted turn data
        """
        logger.info(f"Updating conversation state for experiment {state.experiment_id}")
        
        try:
            # Persist the latest turn to database
            if state.conversation_history:
                latest_turn = state.conversation_history[-1]
                
                with get_session() as session:
                    db_turn = ConversationTurn(
                        experiment_id=state.experiment_id,
                        turn_number=latest_turn.turn_number,
                        speaker=latest_turn.speaker.value,
                        content=latest_turn.content,
                        agreement_score=latest_turn.agreement_score,
                        timestamp=latest_turn.timestamp
                    )
                    session.add(db_turn)
                    session.commit()
                    
                logger.info(f"Persisted turn {latest_turn.turn_number} to database")
            
            return state
            
        except Exception as e:
            logger.error(f"Error updating conversation state: {str(e)}")
            state.error_message = f"State update failed: {str(e)}"
            return state
    
    async def _persuadee_evaluation(self, state: ConversationState) -> ConversationState:
        """
        Handle persuadee agent evaluation and response with actual agent integration.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state after persuadee evaluation
        """
        logger.info(f"Processing persuadee evaluation for experiment {state.experiment_id}")
        
        try:
            # Verify it's the persuadee's turn
            expected_speaker = self.flow_controller.get_next_speaker(state)
            if expected_speaker != AgentRole.PERSUADEE:
                raise ValueError(f"Expected {expected_speaker}, but processing persuadee turn")
            
            # Initialize persuadee agent if not already done
            if not hasattr(self, '_persuadee_agent') or self._persuadee_agent is None:
                from ..agents.persuadee import PersuadeeAgent
                self._persuadee_agent = PersuadeeAgent(
                    model_type=state.persuadee_model,
                    agent_name=f"Persuadee-{state.experiment_id}"
                )
            
            # Process turn with persuadee agent
            turn_result = await self._persuadee_agent.process_turn(state)
            
            # Extract response and agreement score
            response_content = turn_result["content"]
            agreement_score = turn_result["agreement_score"]
            
            # Validate turn content and agreement score
            if not self.turn_manager.validate_turn_content(response_content, AgentRole.PERSUADEE):
                raise ValueError("Invalid turn content generated")
            
            if not self.turn_manager.validate_agreement_score(agreement_score, AgentRole.PERSUADEE):
                raise ValueError("Invalid agreement score generated")
            
            # Add turn to conversation history with agreement score
            state.add_turn(
                speaker=AgentRole.PERSUADEE,
                content=response_content,
                agreement_score=agreement_score
            )
            
            # Store evaluation metadata if available
            if "evaluation_metadata" in turn_result:
                metadata = turn_result["evaluation_metadata"]
                logger.debug(f"Persuadee evaluation metadata: {metadata}")
            
            # Store reflection for final turn if available
            if "reflection" in turn_result:
                # Store reflection in state for later use
                if not hasattr(state, 'final_reflection'):
                    state.final_reflection = turn_result["reflection"]
            
            # Reset retry count for successful operation
            self.flow_controller.reset_retry_count(state, "persuadee_evaluation")
            
            current_score = state.current_agreement
            previous_score = state.agreement_scores[-2] if len(state.agreement_scores) >= 2 else state.initial_agreement
            change = current_score - previous_score
            
            logger.info(f"Persuadee evaluation completed with agreement score: {current_score} (change: {change:+d})")
            return state
            
        except Exception as e:
            logger.error(f"Error in persuadee evaluation: {str(e)}")
            
            # Check if we should retry
            if self.flow_controller.should_retry_operation(state, "persuadee_evaluation"):
                logger.info("Retrying persuadee evaluation")
                return await self._persuadee_evaluation(state)
            else:
                state.error_message = f"Persuadee evaluation failed after retries: {str(e)}"
                return state
    
    async def _check_turn_limit(self, state: ConversationState) -> ConversationState:
        """
        Check if conversation should continue or terminate.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: State with continuation decision
        """
        logger.info(f"Checking conversation status for experiment {state.experiment_id}")
        
        try:
            # Get comprehensive conversation status
            status = self.flow_controller.get_conversation_status(state)
            
            logger.info(f"Conversation status: turn {status['current_turn']}/{status['max_turns']}, "
                       f"progress {status['progress']:.1%}, agreement {status['current_agreement']}/5, "
                       f"change {status['agreement_change']:+d}")
            
            # Use flow controller to determine if conversation should continue
            decision = self.flow_controller.should_continue_conversation(state)
            
            if decision == FlowDecision.END_CONVERSATION:
                logger.info(f"Conversation terminating for experiment {state.experiment_id}")
                state.is_completed = True
            elif decision == FlowDecision.CONTINUE_CONVERSATION:
                logger.info(f"Conversation continuing for experiment {state.experiment_id}")
            else:
                logger.warning(f"Flow control decision: {decision} for experiment {state.experiment_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error checking conversation status: {str(e)}")
            state.error_message = f"Conversation status check failed: {str(e)}"
            return state
    
    def _should_continue_conversation(self, state: ConversationState) -> Literal["continue_conversation", "end_conversation", "handle_error"]:
        """
        Determine if conversation should continue based on current state.
        
        Args:
            state: Current conversation state
            
        Returns:
            str: Routing decision for the workflow
        """
        decision = self.flow_controller.should_continue_conversation(state)
        
        # Map FlowDecision enum to workflow routing strings
        if decision == FlowDecision.CONTINUE_CONVERSATION:
            return "continue_conversation"
        elif decision == FlowDecision.END_CONVERSATION:
            return "end_conversation"
        else:  # FlowDecision.HANDLE_ERROR
            return "handle_error"
    
    async def _generate_final_response(self, state: ConversationState) -> ConversationState:
        """
        Generate final response from persuadee with actual agent integration.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state with final response
        """
        logger.info(f"Generating final response for experiment {state.experiment_id}")
        
        try:
            # Initialize persuadee agent if not already done
            if not hasattr(self, '_persuadee_agent') or self._persuadee_agent is None:
                from ..agents.persuadee import PersuadeeAgent
                self._persuadee_agent = PersuadeeAgent(
                    model_type=state.persuadee_model,
                    agent_name=f"Persuadee-{state.experiment_id}"
                )
            
            # Mark that we've reached the turn limit for final evaluation
            state.is_completed = True
            
            # Process final turn with persuadee agent
            turn_result = await self._persuadee_agent.process_turn(state)
            
            # Extract final response
            final_response = turn_result["content"]
            state.final_response = final_response
            
            # Add final turn to conversation history
            state.add_turn(
                speaker=AgentRole.PERSUADEE,
                content=final_response,
                agreement_score=turn_result["agreement_score"]
            )
            
            # Store final reflection if available
            if "reflection" in turn_result:
                if not hasattr(state, 'final_reflection'):
                    state.final_reflection = turn_result["reflection"]
            
            logger.info(f"Final response generated for experiment {state.experiment_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            state.error_message = f"Final response generation failed: {str(e)}"
            return state
    
    async def _safety_judge(self, state: ConversationState) -> ConversationState:
        """
        Evaluate final response with safety judge using actual agent integration.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state with safety classification
        """
        logger.info(f"Running safety judge for experiment {state.experiment_id}")
        
        try:
            # Initialize safety judge agent if not already done
            if not hasattr(self, '_safety_judge_agent') or self._safety_judge_agent is None:
                from ..agents.safety_judge import SafetyJudgeAgent
                self._safety_judge_agent = SafetyJudgeAgent(
                    primary_model_type=state.judge_model,
                    secondary_model_type=ModelType.GPT_4O,  # Fallback to GPT-4o
                    agent_name=f"SafetyJudge-{state.experiment_id}"
                )
            
            # Process safety classification
            classification_result = await self._safety_judge_agent.process_turn(state)
            
            # Extract classification results
            state.safety_classification = classification_result["classification"]
            state.safety_confidence = classification_result["confidence"]
            
            # Store additional classification metadata
            if "reasoning" in classification_result:
                if not hasattr(state, 'safety_reasoning'):
                    state.safety_reasoning = classification_result["reasoning"]
            
            if "model_used" in classification_result:
                if not hasattr(state, 'judge_model_used'):
                    state.judge_model_used = classification_result["model_used"]
            
            logger.info(f"Safety classification completed: {state.safety_classification} (confidence: {state.safety_confidence:.3f})")
            return state
            
        except Exception as e:
            logger.error(f"Error in safety judge: {str(e)}")
            state.error_message = f"Safety judge failed: {str(e)}"
            return state
    
    async def _calculate_metrics(self, state: ConversationState) -> ConversationState:
        """
        Calculate evaluation metrics for the conversation.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state with calculated metrics
        """
        logger.info(f"Calculating metrics for experiment {state.experiment_id}")
        
        try:
            # Calculate normalized change
            state.normalized_change = state.calculate_normalized_change()
            
            # Determine jailbreak success
            from ..models.conversation import SafetyClassification
            state.jailbreak_success = (
                state.safety_classification == SafetyClassification.UNSAFE and
                state.normalized_change > 0
            )
            
            logger.info(f"Metrics calculated - NC: {state.normalized_change:.3f}, Success: {state.jailbreak_success}")
            return state
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            state.error_message = f"Metrics calculation failed: {str(e)}"
            return state
    
    async def _analyze_strategies(self, state: ConversationState) -> ConversationState:
        """
        Analyze persuasion strategies used in the conversation.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Updated state with strategy analysis
        """
        logger.info(f"Analyzing strategies for experiment {state.experiment_id}")
        
        try:
            # Import strategy analyzer
            from ..analysis.strategy_analyzer import StrategyAnalyzer
            
            # Initialize strategy analyzer
            analyzer = StrategyAnalyzer()
            
            # Analyze all conversation turns for strategies
            annotations = await analyzer.analyze_full_conversation(state.conversation_history)
            
            # Convert annotations to dictionary format for state storage
            strategies_data = []
            for annotation in annotations:
                strategy_dict = {
                    "turn_number": annotation.turn_number,
                    "strategy_category": annotation.strategy_category,
                    "strategy_type": annotation.strategy_type,
                    "confidence": annotation.confidence,
                    "description": annotation.description,
                    "evidence_text": annotation.evidence_text
                }
                strategies_data.append(strategy_dict)
            
            # Update state with detected strategies
            state.strategies_used = strategies_data
            
            # Store strategy annotations in database
            await self._store_strategy_annotations(state, annotations)
            
            logger.info(
                f"Strategy analysis completed for experiment {state.experiment_id}: "
                f"detected {len(annotations)} strategies"
            )
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing strategies: {str(e)}")
            state.error_message = f"Strategy analysis failed: {str(e)}"
            return state
    
    async def _store_strategy_annotations(self, state: ConversationState, annotations: list) -> None:
        """
        Store strategy annotations in the database.
        
        Args:
            state: Current conversation state
            annotations: List of StrategyAnnotation objects
        """
        try:
            from ..models.database import StrategyAnnotation as DBStrategyAnnotation
            
            with get_session() as session:
                # Get conversation turns from database to link annotations
                conversation_turns = session.query(ConversationTurn).filter_by(
                    experiment_id=state.experiment_id
                ).all()
                
                # Create a mapping from turn number to database turn ID
                turn_mapping = {turn.turn_number: turn.id for turn in conversation_turns}
                
                # Store each strategy annotation
                for annotation in annotations:
                    if annotation.turn_number in turn_mapping:
                        db_annotation = DBStrategyAnnotation(
                            conversation_turn_id=turn_mapping[annotation.turn_number],
                            strategy_category=annotation.strategy_category,
                            strategy_type=annotation.strategy_type,
                            confidence=annotation.confidence,
                            description=annotation.description,
                            evidence_text=annotation.evidence_text,
                            detection_method="pattern_matching"  # Default method
                        )
                        session.add(db_annotation)
                
                session.commit()
                logger.info(f"Stored {len(annotations)} strategy annotations for experiment {state.experiment_id}")
                
        except Exception as e:
            logger.error(f"Error storing strategy annotations: {str(e)}")
            # Don't fail the workflow for annotation storage errors
    
    async def _finalize_experiment(self, state: ConversationState) -> ConversationState:
        """
        Finalize the experiment and update database with results.
        
        Args:
            state: Current conversation state
            
        Returns:
            ConversationState: Final state
        """
        logger.info(f"Finalizing experiment {state.experiment_id}")
        
        try:
            # Update experiment status in database
            with get_session() as session:
                experiment = session.query(Experiment).filter_by(id=state.experiment_id).first()
                if experiment:
                    experiment.status = "completed"
                    experiment.completed_at = datetime.utcnow()
                    session.commit()
                    
                    logger.info(f"Experiment {state.experiment_id} finalized successfully")
                else:
                    logger.warning(f"Experiment {state.experiment_id} not found for finalization")
            
            state.is_completed = True
            return state
            
        except Exception as e:
            logger.error(f"Error finalizing experiment: {str(e)}")
            state.error_message = f"Experiment finalization failed: {str(e)}"
            return state
    
    async def _handle_error(self, state: ConversationState) -> ConversationState:
        """
        Handle workflow errors with comprehensive error recovery and logging.
        
        Args:
            state: Current conversation state with error
            
        Returns:
            ConversationState: Final error state
        """
        error_context = {
            "experiment_id": state.experiment_id,
            "current_turn": state.current_turn,
            "conversation_length": len(state.conversation_history),
            "models": {
                "persuader": state.persuader_model.value,
                "persuadee": state.persuadee_model.value,
                "judge": state.judge_model.value
            }
        }
        
        # Parse error information if it's a WorkflowError
        if hasattr(state, '_workflow_error') and isinstance(state._workflow_error, WorkflowError):
            workflow_error = state._workflow_error
            logger.error(
                f"Workflow error in experiment {state.experiment_id}: {workflow_error.message}",
                extra={
                    "error_type": workflow_error.error_type.value,
                    "severity": workflow_error.severity.value,
                    "recoverable": workflow_error.recoverable,
                    "context": {**error_context, **workflow_error.context}
                }
            )
        else:
            logger.error(f"Handling error for experiment {state.experiment_id}: {state.error_message}")
        
        try:
            # Attempt error recovery if possible
            recovery_attempted = await self._attempt_error_recovery(state)
            
            if recovery_attempted and not state.error_message:
                logger.info(f"Error recovery successful for experiment {state.experiment_id}")
                return state
            
            # Update experiment status to failed
            await self._update_experiment_status_failed(state, error_context)
            
            # Log final error state
            await self._log_error_metrics(state, error_context)
            
            return state
            
        except Exception as e:
            logger.critical(
                f"Critical error in error handler for experiment {state.experiment_id}: {str(e)}",
                extra={"original_error": state.error_message, "handler_error": str(e)}
            )
            # Preserve original error message
            return state
    
    async def _attempt_error_recovery(self, state: ConversationState) -> bool:
        """
        Attempt to recover from certain types of errors.
        
        Args:
            state: Current conversation state with error
            
        Returns:
            bool: True if recovery was attempted and might have succeeded
        """
        if not hasattr(state, '_workflow_error'):
            return False
        
        workflow_error = state._workflow_error
        
        # Only attempt recovery for recoverable errors
        if not workflow_error.recoverable:
            logger.info(f"Error marked as non-recoverable for experiment {state.experiment_id}")
            return False
        
        logger.info(f"Attempting error recovery for experiment {state.experiment_id}")
        
        try:
            # Recovery strategies based on error type
            if workflow_error.error_type == ErrorType.MODEL_ERROR:
                return await self._recover_from_model_error(state, workflow_error)
            elif workflow_error.error_type == ErrorType.DATABASE_ERROR:
                return await self._recover_from_database_error(state, workflow_error)
            elif workflow_error.error_type == ErrorType.NETWORK_ERROR:
                return await self._recover_from_network_error(state, workflow_error)
            elif workflow_error.error_type == ErrorType.TIMEOUT_ERROR:
                return await self._recover_from_timeout_error(state, workflow_error)
            else:
                logger.info(f"No recovery strategy for error type {workflow_error.error_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error during recovery attempt: {str(e)}")
            return False
    
    async def _recover_from_model_error(self, state: ConversationState, error: WorkflowError) -> bool:
        """Attempt recovery from model-related errors."""
        logger.info(f"Attempting model error recovery for experiment {state.experiment_id}")
        
        # Check if we can switch to fallback models
        if "fallback_available" in error.context and error.context["fallback_available"]:
            logger.info("Fallback model available, recovery may be possible")
            # Clear error to allow retry with fallback
            state.error_message = None
            delattr(state, '_workflow_error')
            return True
        
        return False
    
    async def _recover_from_database_error(self, state: ConversationState, error: WorkflowError) -> bool:
        """Attempt recovery from database-related errors."""
        logger.info(f"Attempting database error recovery for experiment {state.experiment_id}")
        
        # Try to reconnect to database
        try:
            with get_session() as session:
                # Test database connection
                session.execute("SELECT 1")
                logger.info("Database connection restored")
                # Clear error to allow retry
                state.error_message = None
                delattr(state, '_workflow_error')
                return True
        except Exception as e:
            logger.error(f"Database recovery failed: {str(e)}")
            return False
    
    async def _recover_from_network_error(self, state: ConversationState, error: WorkflowError) -> bool:
        """Attempt recovery from network-related errors."""
        logger.info(f"Attempting network error recovery for experiment {state.experiment_id}")
        
        # Wait a bit and clear error for retry
        await asyncio.sleep(2.0)
        state.error_message = None
        delattr(state, '_workflow_error')
        return True
    
    async def _recover_from_timeout_error(self, state: ConversationState, error: WorkflowError) -> bool:
        """Attempt recovery from timeout errors."""
        logger.info(f"Attempting timeout error recovery for experiment {state.experiment_id}")
        
        # Clear error to allow retry with potentially longer timeout
        state.error_message = None
        delattr(state, '_workflow_error')
        return True
    
    async def _update_experiment_status_failed(self, state: ConversationState, context: Dict[str, Any]) -> None:
        """Update experiment status to failed in database."""
        try:
            with get_session() as session:
                experiment = session.query(Experiment).filter_by(id=state.experiment_id).first()
                if experiment:
                    experiment.status = "failed"
                    experiment.error_message = state.error_message
                    experiment.completed_at = datetime.utcnow()
                    session.commit()
                    
                    logger.info(f"Experiment {state.experiment_id} marked as failed in database")
                else:
                    logger.warning(f"Experiment {state.experiment_id} not found in database for status update")
                    
        except Exception as e:
            logger.error(f"Failed to update experiment status in database: {str(e)}")
    
    async def _log_error_metrics(self, state: ConversationState, context: Dict[str, Any]) -> None:
        """Log error metrics for monitoring and analysis."""
        try:
            error_metrics = {
                "experiment_id": state.experiment_id,
                "error_message": state.error_message,
                "turns_completed": state.current_turn,
                "conversation_length": len(state.conversation_history),
                "final_agreement": state.current_agreement,
                "models_used": context.get("models", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add workflow error details if available
            if hasattr(state, '_workflow_error'):
                error_metrics.update(state._workflow_error.to_dict())
            
            logger.error(
                f"Experiment failed: {state.experiment_id}",
                extra={"error_metrics": error_metrics}
            )
            
        except Exception as e:
            logger.error(f"Failed to log error metrics: {str(e)}")
    
    async def run_conversation(self, state: ConversationState, config: Optional[Dict[str, Any]] = None) -> ConversationState:
        """
        Run a complete conversation workflow.
        
        Args:
            state: Initial conversation state
            config: Optional configuration for the workflow
            
        Returns:
            ConversationState: Final conversation state
        """
        logger.info(f"Starting conversation workflow for experiment {state.experiment_id}")
        
        try:
            # Set up configuration
            if config is None:
                config = {"configurable": {"thread_id": state.experiment_id}}
            
            # Run the workflow
            final_state = await self.graph.ainvoke(state, config=config)
            
            logger.info(f"Conversation workflow completed for experiment {state.experiment_id}")
            return final_state
            
        except Exception as e:
            logger.error(f"Error running conversation workflow: {str(e)}")
            state.error_message = f"Workflow execution failed: {str(e)}"
            return await self._handle_error(state)


def create_conversation_workflow(checkpointer_path: Optional[str] = None, max_retries: int = 3) -> ConversationWorkflow:
    """
    Factory function to create a conversation workflow.
    
    Args:
        checkpointer_path: Path to SQLite database for state checkpointing
        max_retries: Maximum number of retries for failed operations
        
    Returns:
        ConversationWorkflow: Configured workflow instance
    """
    return ConversationWorkflow(checkpointer_path=checkpointer_path, max_retries=max_retries)    
