import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from src.agents.content_generator import content_generator_node
from src.agents.state import TutorState

@patch('src.agents.content_generator.LapaLLM')
def test_content_generator_node(MockLapaLLM):
    # Setup Mock
    mock_llm_instance = MockLapaLLM.return_value
    
    # Mock generate response (valid JSON string)
    sample_response = {
        "lecture_content": "# Lecture Title\n\nContent here with reference [Book1, стор. 10].",
        "control_questions": ["Question 1?", "Question 2?", "Question 3?"]
    }
    mock_llm_instance.generate.return_value = json.dumps(sample_response)
    
    # Setup State
    state: TutorState = {
        "teacher_query": "Explain photosynthesis",
        "grade": 9,
        "subject": "Biology",
        "matched_pages": [
            {
                "book_id": "Book1", 
                "page_number": 10, 
                "page_text": "Photosynthesis is the process...", 
                "topic_title": "Photosynthesis"
            }
        ],
        "student_profile": {"weak_topics": ["Biology Basics"]}
    }
    
    # Mock file open for system prompt
    with patch("builtins.open", mock_open(read_data="System Prompt Content")):
        # Execute
        result = content_generator_node(state)
    
    # Verify Output Structure
    assert "lecture_content" in result
    assert "control_questions" in result
    assert "sources" in result
    
    # Verify Content
    assert len(result["control_questions"]) == 3
    assert result["lecture_content"] == sample_response["lecture_content"]
    assert "[Book1, стор. 10]" in result["sources"]
    
    # Verify LLM Interaction
    mock_llm_instance.generate.assert_called_once()
    
    # Check arguments passed to LLM
    call_args = mock_llm_instance.generate.call_args
    prompt = call_args[0][0]  # First positional argument
    kwargs = call_args[1]     # Keyword arguments
    
    # Check Prompt Content
    assert "Photosynthesis is the process..." in prompt
    assert "Biology Basics" in prompt
    assert "Book1, стор. 10" in prompt
    assert "System Prompt Content" in kwargs.get("system", "")

def test_content_generator_json_cleaning():
    """Test if the node cleans Markdown code blocks from JSON response"""
    with patch('src.agents.content_generator.LapaLLM') as MockLapaLLM:
        mock_llm = MockLapaLLM.return_value
        
        # Response with markdown code blocks
        raw_json = json.dumps({
            "lecture_content": "Clean content",
            "control_questions": []
        })
        mock_llm.generate.return_value = f"```json\n{raw_json}\n```"
        
        state: TutorState = {
            "teacher_query": "test", 
            "matched_pages": []
        }
        
        with patch("builtins.open", mock_open(read_data="System Prompt")):
            result = content_generator_node(state)
        
        assert result["lecture_content"] == "Clean content"

def test_content_generator_fallback_on_json_error():
    """Test if the node correctly falls back when JSON parsing fails."""
    with patch('src.agents.content_generator.LapaLLM') as MockLapaLLM:
        mock_llm = MockLapaLLM.return_value
        
        # 1. First call returns invalid JSON (unescaped newline)
        # 2. Second call (fallback) returns simple text
        mock_llm.generate.side_effect = [
            '{"lecture_content": "Line 1\nLine 2"}',  # Invalid JSON
            '# Fallback Lecture\n\nContent'            # Fallback response
        ]
        
        state: TutorState = {
            "teacher_query": "test", 
            "matched_pages": []
        }
        
        with patch("builtins.open", mock_open(read_data="System Prompt")):
            result = content_generator_node(state)
        
        # Check that we got the fallback content
        assert result["lecture_content"] == '# Fallback Lecture\n\nContent'
        assert result["control_questions"] == []
        
        # Verify two calls were made
        assert mock_llm.generate.call_count == 2

