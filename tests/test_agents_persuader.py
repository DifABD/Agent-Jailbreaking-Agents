"""
Unit tests for the Persuader agent implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.agents.persuader import PersuaderAgent
from src.models.conversation import ModelType, ConversationState, AgentRole, ConversationTurn


class TestPersuaderAgent:
    """Test PersuaderAgent functionality."""
    
    @patch('src.agents.base.get_settings')
    def test_initialization(self, mock_get_settings):
        """Test PersuaderAgent initialization."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model_temperature = 0.7
        mock_settings.model_max_tokens = 1000
        mock_settings.model_timeout = 30
        mock_settings.openai_api_key = "test-key"
        mock_get_settings.return_value = mock_settings
        
        with patch('src.agents.base.ChatOpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            assert agent.model_type == ModelType.GPT_4O
            assert agent.agent_name == "Persuader"
            assert hasattr(agent, 'system_prompt')
            assert hasattr(agent, 'response_template')
            assert hasattr(agent, 'initial_template')
    
    def test_get_system_prompt(self):
        """Test system prompt retrieval."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            prompt = agent.get_system_prompt()
            
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert "adversary" in prompt.lower()
            assert "persuasive" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_process_turn_initial(self):
        """Test processing the initial turn."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            # Mock the model call
            with patch.object(agent, '_call_model', return_value="Initial persuasive argument"):
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
                
                assert result["content"] == "Initial persuasive argument"
                assert result["agent_role"] == AgentRole.PERSUADER
                assert result["turn_number"] == 0
                assert result["model_type"] == "gpt-4o"
                assert "strategy_hints" in result
    
    @pytest.mark.asyncio
    async def test_process_turn_follow_up(self):
        """Test processing a follow-up turn."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            # Mock the model call
            with patch.object(agent, '_call_model', return_value="Follow-up argument"):
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
                            content="Initial argument"
                        ),
                        ConversationTurn(
                            turn_number=1,
                            speaker=AgentRole.PERSUADEE,
                            content="Counter response",
                            agreement_score=2
                        )
                    ],
                    current_turn=2,
                    initial_agreement=1,
                    current_agreement=2,
                    agreement_scores=[2]
                )
                
                result = await agent.process_turn(state)
                
                assert result["content"] == "Follow-up argument"
                assert result["agent_role"] == AgentRole.PERSUADER
                assert result["turn_number"] == 2
    
    @pytest.mark.asyncio
    async def test_generate_initial_response(self):
        """Test initial response generation."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            with patch.object(agent, '_call_model', return_value="  Initial response  "):
                state = ConversationState(
                    experiment_id="test-exp",
                    persuader_model=ModelType.GPT_4O,
                    persuadee_model=ModelType.GPT_4O,
                    judge_model=ModelType.LLAMA_GUARD_2_8B,
                    original_claim="Test claim",
                    jailbreak_goal="Test goal",
                    initial_agreement=1
                )
                
                response = await agent._generate_initial_response(state)
                
                assert response == "Initial response"  # Should be stripped
                agent._call_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_follow_up_response(self):
        """Test follow-up response generation."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            with patch.object(agent, '_call_model', return_value="Follow-up response"):
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
                            content="Previous argument"
                        )
                    ],
                    current_turn=1,
                    current_agreement=1,  # No persuadee turns yet
                    agreement_scores=[]   # No persuadee turns yet
                )
                
                response = await agent._generate_follow_up_response(state)
                
                assert response == "Follow-up response"
                agent._call_model.assert_called_once()
    
    def test_format_conversation_history_empty(self):
        """Test formatting empty conversation history."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[]
            )
            
            formatted = agent._format_conversation_history(state)
            assert formatted == "No previous conversation."
    
    def test_format_conversation_history_with_turns(self):
        """Test formatting conversation history with turns."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
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
                        content="Persuader argument"
                    ),
                    ConversationTurn(
                        turn_number=1,
                        speaker=AgentRole.PERSUADEE,
                        content="Persuadee response",
                        agreement_score=3
                    )
                ],
                current_turn=2,  # Must match history length
                current_agreement=3,
                agreement_scores=[3]  # Must match persuadee turns
            )
            
            formatted = agent._format_conversation_history(state)
            
            assert "You: Persuader argument" in formatted
            assert "Other Agent: Persuadee response" in formatted
            assert "[Agreement Level: 3/5]" in formatted
    
    def test_analyze_response_strategy(self):
        """Test response strategy analysis."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            # Test logical reasoning detection
            response1 = "Because research shows that this is true, therefore we should accept it."
            strategies1 = agent._analyze_response_strategy(response1)
            assert "logical_reasoning" in strategies1
            
            # Test emotional appeal detection
            response2 = "Imagine how you would feel if this happened to you."
            strategies2 = agent._analyze_response_strategy(response2)
            assert "emotional_appeal" in strategies2
            
            # Test authority detection
            response3 = "Experts and scientists agree on this point."
            strategies3 = agent._analyze_response_strategy(response3)
            assert "authority" in strategies3
            
            # Test default case
            response4 = "This is a simple statement."
            strategies4 = agent._analyze_response_strategy(response4)
            assert strategies4 == ["general_persuasion"]
    
    def test_get_persuasion_context(self):
        """Test persuasion context calculation."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            # Test positive progress
            state = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Arg1"),
                    ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="Resp1", agreement_score=3)
                ],
                current_turn=2,
                initial_agreement=1,
                current_agreement=3,
                agreement_scores=[3],  # Must match persuadee turns
                max_turns=7
            )
            
            context = agent.get_persuasion_context(state)
            
            assert context["initial_agreement"] == 1
            assert context["current_agreement"] == 3
            assert context["progress"] == 2
            assert context["effectiveness"] == "positive"
            assert context["turns_taken"] == 1
            assert context["remaining_turns"] == 5
    
    @pytest.mark.asyncio
    async def test_suggest_strategy(self):
        """Test strategy suggestion."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
            # Test positive effectiveness
            state_positive = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Arg1"),
                    ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="Resp1", agreement_score=3)
                ],
                initial_agreement=1,
                current_agreement=3,
                current_turn=2,
                agreement_scores=[3],
                max_turns=7
            )
            
            suggestion = await agent.suggest_strategy(state_positive)
            assert "continue" in suggestion.lower()
            
            # Test negative effectiveness
            state_negative = ConversationState(
                experiment_id="test-exp",
                persuader_model=ModelType.GPT_4O,
                persuadee_model=ModelType.GPT_4O,
                judge_model=ModelType.LLAMA_GUARD_2_8B,
                original_claim="Test claim",
                jailbreak_goal="Test goal",
                conversation_history=[
                    ConversationTurn(turn_number=0, speaker=AgentRole.PERSUADER, content="Arg1"),
                    ConversationTurn(turn_number=1, speaker=AgentRole.PERSUADEE, content="Resp1", agreement_score=1)
                ],
                initial_agreement=3,
                current_agreement=1,
                current_turn=2,
                agreement_scores=[1],
                max_turns=7
            )
            
            suggestion = await agent.suggest_strategy(state_negative)
            assert "different" in suggestion.lower()
            
            # Test final turns
            state_final = ConversationState(
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
                    ConversationTurn(turn_number=3, speaker=AgentRole.PERSUADEE, content="Resp2", agreement_score=2),
                    ConversationTurn(turn_number=4, speaker=AgentRole.PERSUADER, content="Arg3"),
                    ConversationTurn(turn_number=5, speaker=AgentRole.PERSUADEE, content="Resp3", agreement_score=2)
                ],
                initial_agreement=2,
                current_agreement=2,
                current_turn=6,
                agreement_scores=[2, 2, 2],
                max_turns=7
            )
            
            suggestion = await agent.suggest_strategy(state_final)
            assert "final" in suggestion.lower()
    
    @pytest.mark.asyncio
    async def test_process_turn_error_handling(self):
        """Test error handling in process_turn."""
        with patch('src.agents.base.get_settings'), \
             patch('src.agents.base.ChatOpenAI'):
            
            agent = PersuaderAgent(ModelType.GPT_4O)
            
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