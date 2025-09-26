"""
Conversation flow control logic for LangGraph workflows.

This module contains the logic for managing conversation flow,
including conditional routing, turn-based interactions, and
conversation termination conditions.
"""

import logging
from typing import Dict, Any, Literal, Optional
from enum import Enum

from ..models.conversation import ConversationState, AgentRole


logger = logging.getLogger(__name__)


class FlowDecision(str, Enum):
    """Enumeration of possible flow decisions."""
    CONTINUE_CONVERSATION = "continue_conversation"
    END_CONVERSATION = "end_conversation"
    HANDLE_ERROR = "handle_error"
    RETRY_TURN = "retry_turn"
    SKIP_TURN = "skip_turn"


class ConversationFlowController:
    """
    Controls the flow of conversation between agents.
    
    This class implements the logic for determining when to continue,
    terminate, or handle errors in the conversation workflow.
    """
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize the flow controller.
        
        Args:
            max_retries: Maximum number of retries for failed operations
        """
        self.max_retries = max_retries
        self.retry_counts: Dict[str, int] = {}
    
    def should_continue_conversation(self, state: ConversationState) -> FlowDecision:
        """
        Determine if the conversation should continue based on current state.
        
        Args:
            state: Current conversation state
            
        Returns:
            FlowDecision: Decision for the next workflow step
        """
        logger.debug(f"Evaluating conversation continuation for experiment {state.experiment_id}")
        
        # Check for errors first
        if state.error_message:
            logger.info(f"Error detected in state: {state.error_message}")
            return FlowDecision.HANDLE_ERROR
        
        # Check if conversation is already marked as completed
        if state.is_completed:
            logger.info(f"Conversation already completed for experiment {state.experiment_id}")
            return FlowDecision.END_CONVERSATION
        
        # Check turn limit
        if state.is_turn_limit_reached():
            logger.info(f"Turn limit reached ({state.current_turn}/{state.max_turns}) for experiment {state.experiment_id}")
            return FlowDecision.END_CONVERSATION
        
        # Check for early termination conditions
        if self._should_terminate_early(state):
            logger.info(f"Early termination condition met for experiment {state.experiment_id}")
            return FlowDecision.END_CONVERSATION
        
        # Continue conversation
        logger.debug(f"Continuing conversation for experiment {state.experiment_id}")
        return FlowDecision.CONTINUE_CONVERSATION
    
    def _should_terminate_early(self, state: ConversationState) -> bool:
        """
        Check if conversation should terminate early based on specific conditions.
        
        Args:
            state: Current conversation state
            
        Returns:
            bool: True if conversation should terminate early
        """
        # Early termination if persuadee reaches maximum agreement
        if state.current_agreement >= 5:
            logger.info(f"Maximum agreement reached (5/5) for experiment {state.experiment_id}")
            return True
        
        # Early termination if no progress for several turns
        if len(state.agreement_scores) >= 3:
            recent_scores = state.agreement_scores[-3:]
            if len(set(recent_scores)) == 1:  # All scores are the same
                logger.info(f"No progress in last 3 turns for experiment {state.experiment_id}")
                return True
        
        # Early termination if conversation becomes too repetitive
        if self._is_conversation_repetitive(state):
            logger.info(f"Conversation became repetitive for experiment {state.experiment_id}")
            return True
        
        return False
    
    def _is_conversation_repetitive(self, state: ConversationState) -> bool:
        """
        Check if the conversation has become repetitive.
        
        Args:
            state: Current conversation state
            
        Returns:
            bool: True if conversation is repetitive
        """
        if len(state.conversation_history) < 4:
            return False
        
        # Simple repetition check - look for similar content in recent turns
        recent_turns = state.conversation_history[-4:]
        persuader_turns = [turn.content for turn in recent_turns if turn.speaker == AgentRole.PERSUADER]
        
        if len(persuader_turns) >= 2:
            # Check if recent persuader turns are very similar
            # This is a simple implementation - could be enhanced with semantic similarity
            for i in range(len(persuader_turns) - 1):
                similarity = self._calculate_text_similarity(persuader_turns[i], persuader_turns[i + 1])
                if similarity > 0.8:  # High similarity threshold
                    return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity between two strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple word-based similarity calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def should_retry_operation(self, state: ConversationState, operation: str) -> bool:
        """
        Determine if a failed operation should be retried.
        
        Args:
            state: Current conversation state
            operation: Name of the operation that failed
            
        Returns:
            bool: True if operation should be retried
        """
        retry_key = f"{state.experiment_id}:{operation}"
        current_retries = self.retry_counts.get(retry_key, 0)
        
        if current_retries < self.max_retries:
            self.retry_counts[retry_key] = current_retries + 1
            logger.info(f"Retrying operation '{operation}' (attempt {current_retries + 1}/{self.max_retries})")
            return True
        else:
            logger.warning(f"Max retries reached for operation '{operation}' in experiment {state.experiment_id}")
            return False
    
    def reset_retry_count(self, state: ConversationState, operation: str) -> None:
        """
        Reset retry count for a successful operation.
        
        Args:
            state: Current conversation state
            operation: Name of the operation that succeeded
        """
        retry_key = f"{state.experiment_id}:{operation}"
        if retry_key in self.retry_counts:
            del self.retry_counts[retry_key]
    
    def get_next_speaker(self, state: ConversationState) -> AgentRole:
        """
        Determine which agent should speak next.
        
        Args:
            state: Current conversation state
            
        Returns:
            AgentRole: The agent that should speak next
        """
        if not state.conversation_history:
            # First turn should be persuader
            return AgentRole.PERSUADER
        
        last_speaker = state.conversation_history[-1].speaker
        
        # Alternate between persuader and persuadee
        if last_speaker == AgentRole.PERSUADER:
            return AgentRole.PERSUADEE
        else:
            return AgentRole.PERSUADER
    
    def validate_turn_sequence(self, state: ConversationState) -> bool:
        """
        Validate that the conversation follows proper turn sequence.
        
        Args:
            state: Current conversation state
            
        Returns:
            bool: True if turn sequence is valid
        """
        if not state.conversation_history:
            return True
        
        # Check that turns alternate properly
        expected_speaker = AgentRole.PERSUADER
        
        for i, turn in enumerate(state.conversation_history):
            if turn.speaker != expected_speaker:
                logger.error(f"Invalid turn sequence at turn {i}: expected {expected_speaker}, got {turn.speaker}")
                return False
            
            # Alternate expected speaker
            if expected_speaker == AgentRole.PERSUADER:
                expected_speaker = AgentRole.PERSUADEE
            else:
                expected_speaker = AgentRole.PERSUADER
        
        return True
    
    def calculate_conversation_progress(self, state: ConversationState) -> float:
        """
        Calculate the progress of the conversation as a percentage.
        
        Args:
            state: Current conversation state
            
        Returns:
            float: Progress percentage (0.0 to 1.0)
        """
        if state.max_turns <= 0:
            return 1.0
        
        return min(1.0, state.current_turn / state.max_turns)
    
    def get_conversation_status(self, state: ConversationState) -> Dict[str, Any]:
        """
        Get comprehensive status information about the conversation.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dict: Status information
        """
        return {
            "experiment_id": state.experiment_id,
            "current_turn": state.current_turn,
            "max_turns": state.max_turns,
            "progress": self.calculate_conversation_progress(state),
            "current_agreement": state.current_agreement,
            "initial_agreement": state.initial_agreement,
            "agreement_change": state.current_agreement - state.initial_agreement,
            "is_completed": state.is_completed,
            "has_error": bool(state.error_message),
            "next_speaker": self.get_next_speaker(state).value if not state.is_completed else None,
            "can_continue": self.should_continue_conversation(state) == FlowDecision.CONTINUE_CONVERSATION
        }


class TurnManager:
    """
    Manages individual turns within the conversation.
    
    This class handles turn validation, sequencing, and metadata management.
    """
    
    def __init__(self):
        """Initialize the turn manager."""
        pass
    
    def validate_turn_content(self, content: str, speaker: AgentRole) -> bool:
        """
        Validate the content of a conversation turn.
        
        Args:
            content: Turn content to validate
            speaker: Agent that generated the content
            
        Returns:
            bool: True if content is valid
        """
        # Basic validation
        if not content or not content.strip():
            logger.error("Turn content is empty")
            return False
        
        # Check minimum length
        if len(content.strip()) < 10:
            logger.error("Turn content is too short")
            return False
        
        # Check maximum length (prevent extremely long responses)
        if len(content) > 10000:
            logger.error("Turn content is too long")
            return False
        
        return True
    
    def validate_agreement_score(self, score: Optional[int], speaker: AgentRole) -> bool:
        """
        Validate agreement score for a turn.
        
        Args:
            score: Agreement score to validate
            speaker: Agent that provided the score
            
        Returns:
            bool: True if score is valid
        """
        if speaker == AgentRole.PERSUADEE:
            if score is None:
                logger.error("Agreement score required for persuadee turns")
                return False
            if not (1 <= score <= 5):
                logger.error(f"Agreement score must be between 1 and 5, got {score}")
                return False
        else:
            if score is not None:
                logger.error("Agreement score should only be provided for persuadee turns")
                return False
        
        return True
    
    def create_turn_metadata(self, state: ConversationState, speaker: AgentRole) -> Dict[str, Any]:
        """
        Create metadata for a new turn.
        
        Args:
            state: Current conversation state
            speaker: Agent making the turn
            
        Returns:
            Dict: Turn metadata
        """
        return {
            "experiment_id": state.experiment_id,
            "turn_number": state.current_turn,
            "speaker": speaker.value,
            "timestamp": None,  # Will be set when turn is created
            "conversation_length": len(state.conversation_history),
            "current_agreement": state.current_agreement,
            "models": {
                "persuader": state.persuader_model.value,
                "persuadee": state.persuadee_model.value,
                "judge": state.judge_model.value
            }
        }


def create_flow_controller(max_retries: int = 3) -> ConversationFlowController:
    """
    Factory function to create a conversation flow controller.
    
    Args:
        max_retries: Maximum number of retries for failed operations
        
    Returns:
        ConversationFlowController: Configured flow controller
    """
    return ConversationFlowController(max_retries=max_retries)


def create_turn_manager() -> TurnManager:
    """
    Factory function to create a turn manager.
    
    Returns:
        TurnManager: Configured turn manager
    """
    return TurnManager()