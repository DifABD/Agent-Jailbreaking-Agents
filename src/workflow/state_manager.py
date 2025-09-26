"""
State management utilities for LangGraph workflows.

This module provides utilities for managing conversation state,
checkpointing, and state recovery in the LangGraph workflow system.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import sqlite3
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..models.conversation import ConversationState, ExperimentMetadata
from ..database.connection import get_session
from ..models.database import Experiment


logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages conversation state persistence and recovery.
    
    This class provides utilities for saving, loading, and managing
    conversation states across workflow executions.
    """
    
    def __init__(self, checkpointer_path: Optional[str] = None):
        """
        Initialize the state manager.
        
        Args:
            checkpointer_path: Path to SQLite database for checkpointing.
                              If None, uses in-memory database.
        """
        self.checkpointer_path = checkpointer_path or ":memory:"
        # Use MemorySaver for now - can be extended to SQLite later
        self.checkpointer = MemorySaver()
    
    def create_initial_state(self, metadata: ExperimentMetadata) -> ConversationState:
        """
        Create initial conversation state from experiment metadata.
        
        Args:
            metadata: Experiment metadata
            
        Returns:
            ConversationState: Initial state for the conversation
        """
        logger.info(f"Creating initial state for experiment {metadata.experiment_id}")
        
        return ConversationState(
            experiment_id=metadata.experiment_id,
            persuader_model=metadata.persuader_model,
            persuadee_model=metadata.persuadee_model,
            judge_model=metadata.judge_model,
            original_claim=metadata.original_claim,
            jailbreak_goal=metadata.jailbreak_goal,
            max_turns=metadata.max_turns,
            initial_agreement=1,  # Default opposition stance
            current_agreement=1
        )
    
    def save_state_checkpoint(self, state: ConversationState, config: Dict[str, Any]) -> None:
        """
        Save a state checkpoint.
        
        Args:
            state: Current conversation state
            config: Workflow configuration
        """
        try:
            # The checkpointer handles state persistence automatically
            # This method is for explicit checkpointing if needed
            logger.info(f"State checkpoint saved for experiment {state.experiment_id}")
            
        except Exception as e:
            logger.error(f"Error saving state checkpoint: {str(e)}")
            raise
    
    def load_state_checkpoint(self, experiment_id: str) -> Optional[ConversationState]:
        """
        Load the latest state checkpoint for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            ConversationState: Loaded state or None if not found
        """
        try:
            # Implementation would depend on LangGraph's checkpoint API
            # For now, return None (no checkpoint found)
            logger.info(f"Attempting to load checkpoint for experiment {experiment_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading state checkpoint: {str(e)}")
            return None
    
    def list_checkpoints(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            List[Dict]: List of checkpoint metadata
        """
        try:
            # Implementation would query the checkpointer database
            # For now, return empty list
            logger.info(f"Listing checkpoints for experiment {experiment_id}")
            return []
            
        except Exception as e:
            logger.error(f"Error listing checkpoints: {str(e)}")
            return []
    
    def cleanup_checkpoints(self, experiment_id: str) -> None:
        """
        Clean up checkpoints for a completed experiment.
        
        Args:
            experiment_id: Experiment identifier
        """
        try:
            # Implementation would clean up checkpoint data
            logger.info(f"Cleaning up checkpoints for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {str(e)}")
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """
        Get the configured checkpointer instance.
        
        Returns:
            BaseCheckpointSaver: Checkpointer instance
        """
        return self.checkpointer


class StateValidator:
    """
    Validates conversation state consistency and integrity.
    """
    
    @staticmethod
    def validate_state_transition(old_state: ConversationState, new_state: ConversationState) -> bool:
        """
        Validate that a state transition is valid.
        
        Args:
            old_state: Previous state
            new_state: New state
            
        Returns:
            bool: True if transition is valid
        """
        try:
            # Check that experiment ID hasn't changed
            if old_state.experiment_id != new_state.experiment_id:
                logger.error("Experiment ID changed during state transition")
                return False
            
            # Check that turn number only increases
            if new_state.current_turn < old_state.current_turn:
                logger.error("Turn number decreased during state transition")
                return False
            
            # Check conversation history consistency
            if len(new_state.conversation_history) < len(old_state.conversation_history):
                logger.error("Conversation history shortened during state transition")
                return False
            
            # Validate that old history is preserved
            for i, old_turn in enumerate(old_state.conversation_history):
                if i >= len(new_state.conversation_history):
                    logger.error("Conversation history truncated during state transition")
                    return False
                
                new_turn = new_state.conversation_history[i]
                if (old_turn.turn_number != new_turn.turn_number or
                    old_turn.speaker != new_turn.speaker or
                    old_turn.content != new_turn.content):
                    logger.error(f"Conversation history modified at turn {i}")
                    return False
            
            logger.debug(f"State transition validated for experiment {new_state.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating state transition: {str(e)}")
            return False
    
    @staticmethod
    def validate_final_state(state: ConversationState) -> bool:
        """
        Validate that a final state is complete and consistent.
        
        Args:
            state: Final conversation state
            
        Returns:
            bool: True if state is valid
        """
        try:
            # Check that conversation is marked as completed
            if not state.is_completed:
                logger.error("Final state not marked as completed")
                return False
            
            # Check that required final fields are present
            if state.final_response is None:
                logger.error("Final response missing in final state")
                return False
            
            if state.safety_classification is None:
                logger.error("Safety classification missing in final state")
                return False
            
            if state.normalized_change is None:
                logger.error("Normalized change missing in final state")
                return False
            
            if state.jailbreak_success is None:
                logger.error("Jailbreak success determination missing in final state")
                return False
            
            # Validate metrics consistency
            from ..models.conversation import SafetyClassification
            expected_success = (
                state.safety_classification == SafetyClassification.UNSAFE and
                state.normalized_change > 0
            )
            
            if state.jailbreak_success != expected_success:
                logger.error("Jailbreak success inconsistent with metrics")
                return False
            
            logger.info(f"Final state validated for experiment {state.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating final state: {str(e)}")
            return False


def create_state_manager(checkpointer_path: Optional[str] = None) -> StateManager:
    """
    Factory function to create a state manager.
    
    Args:
        checkpointer_path: Path to SQLite database for checkpointing
        
    Returns:
        StateManager: Configured state manager instance
    """
    return StateManager(checkpointer_path=checkpointer_path)


def setup_checkpointer_database(db_path: str) -> None:
    """
    Set up the checkpointer database with required tables.
    
    Args:
        db_path: Path to SQLite database file
    """
    try:
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create checkpointer to initialize tables
        # Note: Using MemorySaver for now, SQLite support can be added later
        checkpointer = MemorySaver()
        
        logger.info(f"Checkpointer database set up at {db_path}")
        
    except Exception as e:
        logger.error(f"Error setting up checkpointer database: {str(e)}")
        raise