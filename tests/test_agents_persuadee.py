"""
Unit tests for the Persuadee agent implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.agents.persuadee import PersuadeeAgent
from src.models.conversation import ModelType, ConversationState, AgentRole, ConversationTurn


class TestPersuadeeAgent:
    """Test PersuadeeAgent functionality."""
    
    @patch('src.agents.base.get_settings')
    def test_initialization(self, mock_get_settings):
        """Test PersuadeeAgent initialization."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model_temperature = 0.7
        mock_settings.model_max_tokens = 1000
        mock_settings.model_timeout = 30
        mock_settings.openai_api_key = "test-key"
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            assert agent.model_type == ModelType.GPT_4O
            assert agent.agent_name == "Persuadee"
            assert hasattr(agent, 'system_prompt')
            assert hasattr(agent, 'initial_template')
            assert hasattr(agent, 'evaluation_template')
            assert hasattr(agent, 'final_template')
    
    def test_get_system_prompt(self):
        """Test system prompt retrieval."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            prompt = agent.get_system_prompt()
            
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert "sensible" in prompt.lower()
            assert "evaluate" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_process_turn_initial(self):
        """Test processing the initial turn (claim evaluation)."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            # Mock the model call
            with patch.object(agent, '_call_model', return_value="RESPONSE: I disagree with this claim. AGREEMENT_SCORE: 2"):
                # Create initial conversation state
                state = ConversationState(
                    experiment_id="test-exp",
                    persuader_model=ModelType.GPT_4O,
                    persuadee_model=ModelType.GPT_4O,
                    judge_model=ModelType.LLAMA_GUARD_2_8B,
                    original_claim="Test claim",
                    jailbreak_goal="Test goal",
                    conversation_history=[],
                    current_turn=0,
                    initial_agreement=1,
                    current_agreement=1
                )
                
                result = await agent.process_turn(state)
                
                assert result["content"] == "I disagree with this claim."
                assert result["agreement_score"] == 2
                assert result["agent_role"] == AgentRole.PERSUADEE
                assert result["turn_number"] == 0
                assert result["model_type"] == "gpt-4o"
                assert "evaluation_metadata" in result
    
    @pytest.mark.asyncio
    async def test_process_turn_evaluation(self):
        """Test processing a persuasion evaluation turn."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            # Mock the model call
            with patch.object(agent, '_call_model', return_value="RESPONSE: That's an interesting point. AGREEMENT_SCORE: 3"):
                # Create conversation state with history
                state = ConversationState(
                    experiment_id="test-exp",
                    persuader_model=ModelType.GPT_4O,
                    persuadee_model=ModelType.GPT_4O,
                    judge_model=ModelType.LLAMA_GUARD_2_8B,
                    original_claim="Test claim",
                    jailbreak_goal="Test goal",
                    conversation_history=[
                        ConversationTurn(
                            turn_number=0,
                            speaker=AgentRole.PERSUADER,
                            content="Persuasive argument"
                        )
                    ],
                    current_turn=1,
                    initial_agreement=2,
                    current_agreement=2,
                    agreement_scores=[],
                    max_turns=7
                )
                
                result = await agent.process_turn(state)
                
                assert result["content"] == "That's an interesting point."
                assert result["agreement_score"] == 3
                assert result["agent_role"] == AgentRole.PERSUADEE
    
    @pytest.mark.asyncio
    async def test_process_turn_final(self):
        """Test processing the final turn."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            # Mock the model call
            with patch.object(agent, '_call_model', return_value="RESPONSE: Final position. AGREEMENT_SCORE: 4 REFLECTION: The conversation was enlightening."):
                # Create conversation state at turn limit
                state = ConversationState(
                    experiment_id="test-exp",
                    persuader_model=ModelType.GPT_4O,
                    persuadee_model=ModelType.GPT_4O,
                    judge_model=ModelType.LLAMA_GUARD_2_8B,
                    original_claim="Test claim",
                    jailbreak_goal="Test goal",
                    conversation_history=[
                        ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Arg1"),
                        ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="Resp1", agreement_score=2),
                        ConversationTurn(turn_number=2, speaker=AgentRole.PERSUADER, content="Arg2"),
                        ConversationTurn(turn_number=3, speaker=AgentRole.PERSUADEE, content="Resp2", agreement_score=3),
                        ConversationTurn(turn_number=4, speaker=AgentRole.PERSUADER, content="Arg3")
                    ],
                    current_turn=5,
                    initial_agreement=2,
                    current_agreement=3,
                    agreement_scores=[2, 3],
                    max_turns=5  # At limit
                )
                
                result = await agent.process_turn(state)
                
                assert result["content"] == "Final position."
                assert result["agreement_score"] == 4
                assert "reflection" in result
    
    @pytest.mark.asyncio
    async def test_generate_initial_evaluation(self):
        """Test initial evaluation generation."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            with patch.object(agent, '_call_model', return_value="RESPONSE: Initial thoughts. AGREEMENT_SCORE: 1"):
                state = ConversationState(
                    experiment_id="test-exp",
                    persuader_model=ModelType.GPT_4O,
                    persuadee_model=ModelType.GPT_4O,
                    judge_model=ModelType.LLAMA_GUARD_2_8B,
                    original_claim="Test claim",
                    jailbreak_goal="Test goal"
                )
                
                response, score = await agent._generate_initial_evaluation(state)
                
                assert response == "Initial thoughts."
                assert score == 1
                agent._call_model.assert_called_once()
    
    def test_parse_response_with_score(self):
        """Test parsing response with agreement score."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            # Test normal format
            raw_response1 = "RESPONSE: This is my response. AGREEMENT_SCORE: 4"
            response1, score1 = agent._parse_response_with_score(raw_response1)
            assert response1 == "This is my response."
            assert score1 == 4
            
            # Test case insensitive
            raw_response2 = "response: Another response. agreement_score: 2"
            response2, score2 = agent._parse_response_with_score(raw_response2)
            assert response2 == "Another response."
            assert score2 == 2
            
            # Test fallback for missing pattern
            raw_response3 = "Just a plain response with score 5 somewhere"
            response3, score3 = agent._parse_response_with_score(raw_response3)
            assert response3 == "Just a plain response with score 5 somewhere"
            assert score3 == 5
            
            # Test default score when no number found
            raw_response4 = "No score in this response"
            response4, score4 = agent._parse_response_with_score(raw_response4)
            assert response4 == "No score in this response"
            assert score4 == 3  # Default
    
    def test_parse_final_response(self):
        """Test parsing final response with reflection."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            raw_response = "RESPONSE: Final thoughts. AGREEMENT_SCORE: 5 REFLECTION: Very persuasive conversation."
            response, score, reflection = agent._parse_final_response(raw_response)
            
            assert response == "Final thoughts."
            assert score == 5
            assert reflection == "Very persuasive conversation."
    
    def test_get_latest_persuader_argument(self):
        """Test getting the latest persuader argument."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="First argument"),
                    ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="Response", agreement_score=2),
                    ConversationTurn(turn_number=2, speaker=AgentRole.PERSUADER, content="Second argument")
                ],
                current_turn=3,
                agreement_scores=[2]
            )
            
            latest_arg = agent._get_latest_persuader_argument(state)
            assert latest_arg == "Second argument"
    
    def test_get_previous_persuadee_response(self):
        """Test getting the previous persuadee response."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Argument"),
                    ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="My response", agreement_score=2)
                ],
                current_turn=2,
                agreement_scores=[2]
            )
            
            previous_response = agent._get_previous_persuadee_response(state)
            assert previous_response == "My response"
            
            # Test with no previous response
            empty_state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[]
            )
            
            no_response = agent._get_previous_persuadee_response(empty_state)
            assert no_response is None
    
    def test_format_conversation_history(self):
        """Test conversation history formatting."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            # Test empty history
            empty_state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[]
            )
            
            formatted_empty = agent._format_conversation_history(empty_state)
            assert formatted_empty == "No previous conversation."
            
            # Test with conversation
            state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Persuader says this"),
                    ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="I respond", agreement_score=3)
                ],
                current_turn=2,
                agreement_scores=[3]
            )
            
            formatted = agent._format_conversation_history(state)
            assert "Other Agent: Persuader says this" in formatted
            assert "You: I respond" in formatted
            assert "[Your Agreement Level: 3/5]" in formatted
    
    def test_generate_evaluation_metadata(self):
        """Test evaluation metadata generation."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Arg1")
                ],
                current_turn=1,
                initial_agreement=2,
                current_agreement=3
            )
            
            metadata = agent._generate_evaluation_metadata(state, 4)
            
            assert metadata["previous_score"] == 3
            assert metadata["new_score"] == 4
            assert metadata["score_change"] == 1
            assert metadata["trend"] == "increasing"
            assert metadata["initial_score"] == 2
            assert metadata["total_change"] == 2
    
    def test_calculate_persuasion_resistance(self):
        """Test persuasion resistance calculation."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            # Test high resistance (no change)
            stable_state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Arg1"),
                    ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="Resp1", agreement_score=3),
                    ConversationTurn(turn_number=2, speaker=AgentRole.PERSUADER, content="Arg2"),
                    ConversationTurn(turn_number=3, speaker=AgentRole.PERSUADEE, content="Resp2", agreement_score=3),
                    ConversationTurn(turn_number=4, speaker=AgentRole.PERSUADER, content="Arg3"),
                    ConversationTurn(turn_number=5, speaker=AgentRole.PERSUADEE, content="Resp3", agreement_score=3)
                ],
                current_turn=6,
                initial_agreement=3,
                current_agreement=3,
                agreement_scores=[3, 3, 3]
            )
            
            resistance = agent.calculate_persuasion_resistance(stable_state)
            assert resistance["resistance_level"] == "very_high"
            assert resistance["total_change"] == 0
            assert resistance["stability"] > 0.9  # High stability
            
            # Test low resistance (big change)
            changed_state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Arg1"),
                    ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="Resp1", agreement_score=2),
                    ConversationTurn(turn_number=2, speaker=AgentRole.PERSUADER, content="Arg2"),
                    ConversationTurn(turn_number=3, speaker=AgentRole.PERSUADEE, content="Resp2", agreement_score=3),
                    ConversationTurn(turn_number=4, speaker=AgentRole.PERSUADER, content="Arg3"),
                    ConversationTurn(turn_number=5, speaker=AgentRole.PERSUADEE, content="Resp3", agreement_score=4),
                    ConversationTurn(turn_number=6, speaker=AgentRole.PERSUADER, content="Arg4"),
                    ConversationTurn(turn_number=7, speaker=AgentRole.PERSUADEE, content="Resp4", agreement_score=5)
                ],
                current_turn=8,
                initial_agreement=1,
                current_agreement=5,
                agreement_scores=[2, 3, 4, 5]
            )
            
            resistance2 = agent.calculate_persuasion_resistance(changed_state)
            assert resistance2["resistance_level"] == "very_low"
            assert resistance2["total_change"] == 4
    
    @pytest.mark.asyncio
    async def test_explain_position(self):
        """Test position explanation."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            # Test significant change
            state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                initial_agreement=1,
                current_agreement=4
            )
            
            explanation = await agent.explain_position(state)
            
            assert "4/5" in explanation
            assert "1/5" in explanation
            assert "more supportive" in explanation
            assert "Significant" in explanation
    
    @pytest.mark.asyncio
    async def test_process_turn_error_handling(self):
        """Test error handling in process_turn."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuadeeAgent(ModelType.GPT_4O)
            
            # Mock the model call to raise an exception
            with patch.object(agent, '_call_model', side_effect=Exception("Model error")):
                state = ConversationState(
                    experiment_id="test-exp",
                    persuader_model=ModelType.GPT_4O,
                    persuadee_model=ModelType.GPT_4O,
                    judge_model=ModelType.LLAMA_GUARD_2_8B,
                    original_claim="Test claim",
                    jailbreak_goal="Test goal"
                )
                
                with pytest.raises(Exception, match="Model error"):
                    await agent.process_turn(state)