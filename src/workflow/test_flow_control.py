"""
Test for conversation flow control logic.

This module contains tests to verify that the flow control
logic works correctly for various conversation scenarios.
"""

import asyncio
import logging
from uuid import uuid4

from .graph import create_conversation_workflow
from .flow_control import create_flow_controller, FlowDecision
from ..models.conversation import ConversationState, ModelType, AgentRole


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_flow_control_logic():
    """
    Test flow control logic with various conversation states.
    """
    logger.info("Starting flow control logic test")
    
    try:
        # Create flow controller
        flow_controller = create_flow_controller()
        logger.info("Flow controller created successfully")
        
        # Test 1: Empty conversation should continue
        test_state = ConversationState(
            experiment_id=str(uuid4()),
            persuader_model=ModelType.GPT_4O,
            persuadee_model=ModelType.LLAMA_3_3_70B,
            judge_model=ModelType.LLAMA_GUARD_2_8B,
            original_claim="Test claim for flow control",
            jailbreak_goal="Test jailbreak goal",
            max_turns=5
        )
        
        decision = flow_controller.should_continue_conversation(test_state)
        assert decision == FlowDecision.CONTINUE_CONVERSATION, f"Expected CONTINUE, got {decision}"
        logger.info("✓ Empty conversation continues correctly")
        
        # Test 2: Conversation at turn limit should end
        test_state.current_turn = 5
        test_state.max_turns = 5
        decision = flow_controller.should_continue_conversation(test_state)
        assert decision == FlowDecision.END_CONVERSATION, f"Expected END, got {decision}"
        logger.info("✓ Turn limit termination works correctly")
        
        # Test 3: Error state should be handled
        test_state.current_turn = 2
        test_state.error_message = "Test error"
        decision = flow_controller.should_continue_conversation(test_state)
        assert decision == FlowDecision.HANDLE_ERROR, f"Expected ERROR, got {decision}"
        logger.info("✓ Error handling works correctly")
        
        # Test 4: Maximum agreement should terminate early
        test_state.error_message = None
        test_state.current_turn = 2
        test_state.current_agreement = 5
        decision = flow_controller.should_continue_conversation(test_state)
        assert decision == FlowDecision.END_CONVERSATION, f"Expected END for max agreement, got {decision}"
        logger.info("✓ Early termination for maximum agreement works correctly")
        
        # Test 5: Next speaker logic
        test_state.current_agreement = 2
        test_state.conversation_history = []
        next_speaker = flow_controller.get_next_speaker(test_state)
        assert next_speaker == AgentRole.PERSUADER, f"Expected PERSUADER first, got {next_speaker}"
        
        # Add a persuader turn
        test_state.add_turn(AgentRole.PERSUADER, "Test persuader message")
        next_speaker = flow_controller.get_next_speaker(test_state)
        assert next_speaker == AgentRole.PERSUADEE, f"Expected PERSUADEE after persuader, got {next_speaker}"
        logger.info("✓ Next speaker logic works correctly")
        
        # Test 6: Turn sequence validation
        is_valid = flow_controller.validate_turn_sequence(test_state)
        assert is_valid, "Turn sequence should be valid"
        logger.info("✓ Turn sequence validation works correctly")
        
        # Test 7: Conversation progress calculation
        test_state.current_turn = 3
        test_state.max_turns = 6
        progress = flow_controller.calculate_conversation_progress(test_state)
        expected_progress = 3 / 6
        assert abs(progress - expected_progress) < 0.01, f"Expected progress {expected_progress}, got {progress}"
        logger.info("✓ Progress calculation works correctly")
        
        logger.info("All flow control logic tests passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Flow control logic test failed: {str(e)}")
        return False


async def test_enhanced_workflow():
    """
    Test the enhanced workflow with flow control.
    """
    logger.info("Starting enhanced workflow test")
    
    try:
        # Create workflow with flow control
        workflow = create_conversation_workflow(max_retries=2)
        logger.info("Enhanced workflow created successfully")
        
        # Create test state
        test_state = ConversationState(
            experiment_id=str(uuid4()),
            persuader_model=ModelType.GPT_4O,
            persuadee_model=ModelType.LLAMA_3_3_70B,
            judge_model=ModelType.LLAMA_GUARD_2_8B,
            original_claim="Test claim for enhanced workflow",
            jailbreak_goal="Test jailbreak goal",
            max_turns=4  # Short conversation for testing
        )
        
        logger.info(f"Test state created for experiment {test_state.experiment_id}")
        
        # Test workflow components
        assert workflow.flow_controller is not None, "Flow controller should be initialized"
        assert workflow.turn_manager is not None, "Turn manager should be initialized"
        logger.info("✓ Workflow components initialized correctly")
        
        # Test conversation status
        status = workflow.flow_controller.get_conversation_status(test_state)
        assert status["experiment_id"] == test_state.experiment_id
        assert status["current_turn"] == 0
        assert status["progress"] == 0.0
        assert status["can_continue"] == True
        logger.info("✓ Conversation status reporting works correctly")
        
        logger.info("Enhanced workflow test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced workflow test failed: {str(e)}")
        return False


async def run_all_tests():
    """
    Run all flow control tests.
    """
    logger.info("Running all flow control tests")
    
    tests = [
        ("Flow Control Logic", test_flow_control_logic),
        ("Enhanced Workflow", test_enhanced_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        result = await test_func()
        results.append((test_name, result))
        
        if result:
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n--- Test Summary ---")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All flow control tests passed!")
        return True
    else:
        print("❌ Some flow control tests failed")
        return False


if __name__ == "__main__":
    # Run all tests
    result = asyncio.run(run_all_tests())
    if not result:
        exit(1)