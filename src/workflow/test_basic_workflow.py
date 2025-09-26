"""
Basic test for LangGraph workflow functionality.

This module contains a simple test to verify that the workflow
can be created and basic state management works correctly.
"""

import asyncio
import logging
from uuid import uuid4

from .graph import create_conversation_workflow
from .state_manager import create_state_manager
from ..models.conversation import ConversationState, ModelType


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_workflow():
    """
    Test basic workflow creation and initialization.
    """
    logger.info("Starting basic workflow test")
    
    try:
        # Create workflow
        workflow = create_conversation_workflow()
        logger.info("Workflow created successfully")
        
        # Create test state
        test_state = ConversationState(
            experiment_id=str(uuid4()),
            persuader_model=ModelType.GPT_4O,
            persuadee_model=ModelType.LLAMA_3_3_70B,
            judge_model=ModelType.LLAMA_GUARD_2_8B,
            original_claim="Test claim for basic workflow",
            jailbreak_goal="Test jailbreak goal",
            max_turns=3
        )
        
        logger.info(f"Test state created for experiment {test_state.experiment_id}")
        
        # Test state manager
        state_manager = create_state_manager()
        logger.info("State manager created successfully")
        
        # Test basic state operations
        initial_state = state_manager.create_initial_state(test_state)
        logger.info("Initial state created from metadata")
        
        # Verify state properties
        assert initial_state.experiment_id == test_state.experiment_id
        assert initial_state.current_turn == 0
        assert len(initial_state.conversation_history) == 0
        assert initial_state.current_agreement == 1
        
        logger.info("Basic workflow test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Basic workflow test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_basic_workflow())
    if result:
        print("✓ Basic workflow test passed")
    else:
        print("✗ Basic workflow test failed")
        exit(1)