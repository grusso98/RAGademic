import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from memory_manager import _SUMMARY_PROMPT, MemoryManager


# Fixture to mock file operations (open, os.path.exists)
@pytest.fixture
def mock_file_system(mocker):
    mock_exists = mocker.patch('memory_manager.os.path.exists')
    # Use MagicMock for the file handle itself to allow checking methods like write
    mock_file_handle = MagicMock()
    mock_open_patch = mocker.patch('memory_manager.open', new_callable=mock_open)
    mock_open_patch.return_value.__enter__.return_value = mock_file_handle # Ensure __enter__ returns the mock file handle
    return mock_exists, mock_open_patch, mock_file_handle # Return the mock file handle too

# Fixture to mock ollama.chat for summarization
@pytest.fixture
def mock_ollama_chat(mocker):
    mock_chat = mocker.patch('memory_manager.ollama.chat')
    return mock_chat

# Helper to reconstruct the expected prompt text (adjust if _SUMMARY_PROMPT changes)
def _format_summary_prompt(history_text: str) -> str:
    """Helper to format the summary prompt using the template from memory_manager."""
    return _SUMMARY_PROMPT.format(history_text=history_text)

# Test cases for MemoryManager.__init__

def test_memory_manager_init_no_file(mock_file_system, mocker):
    """Tests initialization when the memory file does not exist."""
    mock_exists, _, _ = mock_file_system
    mock_exists.return_value = False # Simulate file not existing

    # Mock summarize_memory during init to isolate __init__ logic flow
    mock_summarize = mocker.patch.object(MemoryManager, 'summarize_memory')

    mem_manager = MemoryManager("non_existent_memory.json")

    mock_exists.assert_called_once_with("non_existent_memory.json")
    assert mem_manager.memory_file == "non_existent_memory.json"
    assert mem_manager.history == []
    assert mem_manager.summary == "No previous conversation history found."
    mock_summarize.assert_not_called() # summarize_memory should not be called if no file

def test_memory_manager_init_with_file(mock_file_system, mocker):
    """Tests initialization when the memory file exists and contains history."""
    mock_exists, mock_open_patch, _ = mock_file_system
    mock_exists.return_value = True # Simulate file existing

    # Simulate file content
    mock_history_data = [["user1", "ai1"], ["user2", "ai2"]]
    mock_open_patch.return_value.__enter__.return_value.read.return_value = json.dumps(mock_history_data)

    # Mock summarize_memory within load_memory to prevent its side effects (like calling ollama)
    # but allow checking if load_memory attempts to call it.
    mock_summarize = mocker.patch.object(MemoryManager, 'summarize_memory')

    mem_manager = MemoryManager("existing_memory.json")

    mock_exists.assert_called_once_with("existing_memory.json")
    mock_open_patch.assert_called_once_with("existing_memory.json", 'r', encoding='utf-8')
    assert mem_manager.history == mock_history_data
    mock_summarize.assert_called_once() # Ensure summarize_memory was called by load_memory

def test_memory_manager_init_json_decode_error(mock_file_system, mocker):
    """Tests initialization when loading the memory file fails due to JSON error."""
    mock_exists, mock_open_patch, _ = mock_file_system
    mock_exists.return_value = True
    mock_open_patch.return_value.__enter__.return_value.read.return_value = "invalid json" # Simulate invalid JSON

    # Mock summarize_memory during init to isolate __init__ logic flow
    mock_summarize = mocker.patch.object(MemoryManager, 'summarize_memory')
    # Mock glog.error to check if it's called
    mock_glog_error = mocker.patch('memory_manager.glog.error')

    mem_manager = MemoryManager("bad_memory.json")

    mock_exists.assert_called_once_with("bad_memory.json")
    mock_open_patch.assert_called_once_with("bad_memory.json", 'r', encoding='utf-8')
    mock_glog_error.assert_called_once() # Ensure error is logged
    assert mem_manager.history == [] # History should be reset
    assert mem_manager.summary == "Error loading conversation history." # Summary should indicate error
    mock_summarize.assert_not_called() # summarize_memory should not be called if load fails

def test_memory_manager_init_io_error_loading(mock_file_system, mocker):
    """Tests initialization when an IOError occurs during file loading."""
    mock_exists, mock_open_patch, _ = mock_file_system
    mock_exists.return_value = True
    mock_open_patch.side_effect = IOError("Permission denied") # Simulate IOError

    mock_summarize = mocker.patch.object(MemoryManager, 'summarize_memory')
    mock_glog_error = mocker.patch('memory_manager.glog.error')

    mem_manager = MemoryManager("io_error_load.json")

    mock_exists.assert_called_once_with("io_error_load.json")
    mock_open_patch.assert_called_once_with("io_error_load.json", 'r', encoding='utf-8')
    mock_glog_error.assert_called_once()
    assert mem_manager.history == []
    assert mem_manager.summary == "Error loading conversation history."
    mock_summarize.assert_not_called()


# Test cases for MemoryManager.save_memory

def test_memory_manager_save_memory(mock_file_system):
    """Tests saving conversation history by checking the written content."""
    mock_exists, mock_open_patch, mock_file_handle = mock_file_system
    # Simulate file existing (or not), load doesn't write initially
    mock_exists.return_value = False
    mem_manager = MemoryManager("save_test.json")
    mem_manager.history = [["u1", "a1"], ["u2", "a2"]]

    mem_manager.save_memory()

    mock_open_patch.assert_called_once_with("save_test.json", 'w', encoding='utf-8')

    # Check that write was called at least once
    mock_file_handle.write.assert_called()

    # Concatenate all content written to the mock file handle
    written_content = "".join(call.args[0] for call in mock_file_handle.write.call_args_list)

    # Verify that the concatenated content is the correct JSON output
    expected_content = json.dumps(mem_manager.history, indent=4, ensure_ascii=False)
    assert written_content == expected_content

def test_memory_manager_save_memory_io_error(mock_file_system, mocker):
    """Tests handling IO error during saving."""
    mock_exists, mock_open_patch, _ = mock_file_system
    mock_exists.return_value = False
    mem_manager = MemoryManager("save_error.json")
    mem_manager.history = [["u1", "a1"]]

    # Simulate IOError during file writing
    mock_open_patch.side_effect = IOError("Disk full")
    mock_glog_error = mocker.patch('memory_manager.glog.error')

    mem_manager.save_memory()

    mock_open_patch.assert_called_once_with("save_error.json", 'w', encoding='utf-8')
    mock_glog_error.assert_called_once() # Ensure error is logged


# Test cases for MemoryManager.add_interaction

def test_memory_manager_add_interaction(mock_file_system, mocker):
    """Tests adding a new interaction."""
    mock_exists, _, _ = mock_file_system
    mock_exists.return_value = False

    # Mock load_memory during init to prevent side effects
    mocker.patch.object(MemoryManager, 'load_memory')

    # Mock save_memory to prevent actual file writing during add_interaction
    # We will assert save_memory was called instead
    mock_save = mocker.patch.object(MemoryManager, 'save_memory')

    # Mock summarize_memory to prevent it from running automatically except when expected
    mock_summarize = mocker.patch.object(MemoryManager, 'summarize_memory')


    mem_manager = MemoryManager("add_test.json")
    initial_history = mem_manager.get_history()
    assert initial_history == []

    # Add 4 interactions - should not trigger summarization
    for i in range(1, 5):
        mem_manager.add_interaction(f"User {i}", f"AI {i}")
        assert len(mem_manager.get_history()) == i
        mock_save.assert_called_once()
        mock_save.reset_mock() # Reset after each save call
        mock_summarize.assert_not_called()

    # Add the 5th interaction - should trigger summarization
    mem_manager.add_interaction("User 5", "AI 5")
    assert len(mem_manager.get_history()) == 5
    mock_save.assert_called_once()
    mock_summarize.assert_called_once()

    # Add more interactions to check if summarization is triggered every 5
    mock_save.reset_mock()
    mock_summarize.reset_mock()
    for i in range(6, 10): # Add interactions 6, 7, 8, 9
        mem_manager.add_interaction(f"User {i}", f"AI {i}")
        assert len(mem_manager.get_history()) == i
        mock_save.assert_called_once()
        mock_save.reset_mock()
        mock_summarize.assert_not_called()

    # Add the 10th interaction - should trigger summarization again
    mem_manager.add_interaction("User 10", "AI 10")
    assert len(mem_manager.get_history()) == 10
    mock_save.assert_called_once()
    mock_summarize.assert_called_once()


# Test cases for MemoryManager.get_history and MemoryManager.get_summary

def test_memory_manager_getters(mock_file_system, mocker):
    """Tests getter methods."""
    mock_exists, mock_open_patch, _ = mock_file_system
    mock_exists.return_value = True
    mock_history_data = [["get_u1", "get_a1"]]
    mock_open_patch.return_value.__enter__.return_value.read.return_value = json.dumps(mock_history_data)

    # Mock summarize_memory during init to control the summary content for this test
    mock_summarize = mocker.patch.object(MemoryManager, 'summarize_memory')

    mem_manager = MemoryManager("getter_test.json")
    mem_manager.summary = "This is a test summary." # Manually set summary after bypassing load

    assert mem_manager.get_history() == mock_history_data
    assert mem_manager.get_summary() == "This is a test summary."
    mock_summarize.assert_called_once() # summarize_memory is called by load_memory (which we mocked) but let's confirm the *intent*

# Test cases for MemoryManager.clear_memory

def test_memory_manager_clear_memory(mock_file_system, mocker):
    """Tests clearing the memory."""
    mock_exists, mock_open_patch, _ = mock_file_system
    mock_exists.return_value = True
    mock_history_data = [["to_clear_u1", "to_clear_a1"]]
    mock_open_patch.return_value.__enter__.return_value.read.return_value = json.dumps(mock_history_data)

    # Mock load_memory during init
    mocker.patch.object(MemoryManager, 'load_memory')
    # Mock save_memory to assert it's called
    mock_save = mocker.patch.object(MemoryManager, 'save_memory')

    mem_manager = MemoryManager("clear_test.json")
    mem_manager.history = mock_history_data # Manually set history as load_memory is mocked
    mem_manager.summary = "Initial summary before clearing." # Manually set summary

    assert mem_manager.get_history() == mock_history_data # Ensure history is as set
    assert mem_manager.get_summary() == "Initial summary before clearing."

    mem_manager.clear_memory()

    assert mem_manager.get_history() == []
    assert mem_manager.get_summary() == "Conversation history has been cleared."
    mock_save.assert_called_once() # Ensure save_memory was called after clearing


# Helper to reconstruct the expected prompt text (adjust if _SUMMARY_PROMPT changes)
# Ensure this helper precisely matches how history is formatted in summarize_memory
def _format_history_text_for_prompt(history: list) -> str:
    """Formats the conversation history into a single string for the prompt."""
    return "\n".join([f"User: {q}\nAI: {a}" for q, a in history])

def _format_summary_prompt(history_text: str) -> str:
    """Helper to format the summary prompt using the template from memory_manager."""
    # Ensure this precisely matches the template usage in summarize_memory
    return _SUMMARY_PROMPT.format(history_text=history_text)


def test_memory_manager_summarize_memory_success(mock_ollama_chat, mocker):
    """Tests successful summarization using Ollama."""
    # Mock load_memory during initialization to prevent side effects
    mocker.patch.object(MemoryManager, 'load_memory')

    mem_manager = MemoryManager(model_identifier="test_model")
    mem_manager.history = [["User: Q1", "AI: A1"], ["User: Q2", "AI: A2"]] # Manually set history

    mock_ollama_chat.return_value = {
        'message': {'content': 'This is the summary of the conversation.'}
    }

    mem_manager.summarize_memory() # Explicitly call summarize_memory

    # Construct the expected history text exactly as the function would
    expected_history_text = _format_history_text_for_prompt(mem_manager.history)

    # Construct the expected full prompt string
    expected_prompt = _format_summary_prompt(expected_history_text)

    mock_ollama_chat.assert_called_once_with(
        model="test_model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": expected_prompt} # Assert against the precisely constructed prompt
        ],
        stream=False
    )
    assert mem_manager.get_summary() == "This is the summary of the conversation."

def test_memory_manager_summarize_memory_no_history():
    """Tests summarization when there is no history."""
    # Mock load_memory during initialization to prevent side effects
    with patch.object(MemoryManager, 'load_memory'):
        mem_manager = MemoryManager()
        mem_manager.history = [] # Ensure history is empty after bypassing load

    # Mock ollama.chat to ensure it's NOT called
    with patch('memory_manager.ollama.chat') as mock_ollama_chat:
         mem_manager.summarize_memory() # Explicitly call summarize_memory
         mock_ollama_chat.assert_not_called()
         assert mem_manager.get_summary() == "No conversation history to summarize."

def test_memory_manager_summarize_memory_long_history(mock_ollama_chat, mocker):
    """Tests summarization with a long history that needs truncation."""
    # Mock load_memory during initialization to prevent side effects
    mocker.patch.object(MemoryManager, 'load_memory')

    mem_manager = MemoryManager(model_identifier="test_model")
    # Create a long history that clearly exceeds the truncation limit (10000 chars)
    long_history_item = ["User: " + "A" * 600, "AI: " + "B" * 600] # Each is ~1200 chars when formatted
    mem_manager.history = [long_history_item] * 10 # 10 interactions = ~12000 chars > 10000

    mock_ollama_chat.return_value = {
        'message': {'content': 'Summary of long history.'}
    }

    mem_manager.summarize_memory() # Explicitly call summarize_memory

    mock_ollama_chat.assert_called_once() # Assert it was called once

    # Check the prompt content for truncation
    called_prompt = mock_ollama_chat.call_args[1]['messages'][1]['content']
    history_text_in_prompt_full = called_prompt.split("Conversation History:\n")[1].split("\nConcise Summary:")[0]
    # Remove the trailing newline from the history text in the prompt for accurate comparison
    history_text_in_prompt = history_text_in_prompt_full.rstrip()


    # Check that the history text in the prompt is truncated
    # Allow a small buffer around the MAX_HISTORY_CHARS for separator characters
    assert len(history_text_in_prompt) <= 10000 + 50 # Relaxing slightly for separator handling

    # More robust check for long history: ensure it contains the end of the history
    # We reconstruct the expected full history text representation
    full_history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in mem_manager.history])

    # The truncated history text in the prompt should be the last _MAX_HISTORY_CHARS of the full history text
    # Note: The truncation happens BEFORE formatting with the prompt template.
    # Let's re-check the source code logic: history_text = history_text[-max_history_chars:]
    # The truncation *is* applied to the joined history text *before* formatting.
    # So we check if the history text *in the prompt* is the end of the *fully joined* history text.

    expected_truncated_text = full_history_text[-10000:]

    assert history_text_in_prompt == expected_truncated_text # Check for exact match after truncation

    assert mem_manager.get_summary() == "Summary of long history."


def test_memory_manager_summarize_memory_ollama_error(mock_ollama_chat, mocker):
    """Tests handling Ollama error during summarization."""
    # Mock load_memory during initialization to prevent side effects
    mocker.patch.object(MemoryManager, 'load_memory')

    mem_manager = MemoryManager(model_identifier="test_model")
    mem_manager.history = [["U", "A"]] # Manually set history

    mock_ollama_chat.side_effect = Exception("Ollama summarize error")
    mock_glog_error = mocker.patch('memory_manager.glog.error')

    mem_manager.summarize_memory() # Explicitly call summarize_memory

    mock_ollama_chat.assert_called_once() # Should be called once despite error
    mock_glog_error.assert_called_once() # Ensure error is logged
    assert mem_manager.get_summary() == "Error occurred during summarization."

def test_memory_manager_summarize_memory_malformed_response(mock_ollama_chat, mocker):
    """Tests handling malformed response from Ollama during summarization."""
    # Mock load_memory during initialization to prevent side effects
    mocker.patch.object(MemoryManager, 'load_memory')

    mem_manager = MemoryManager(model_identifier="test_model")
    mem_manager.history = [["U", "A"]] # Manually set history

    mock_ollama_chat.return_value = {'unexpected_key': 'value'} # Simulate malformed response (Missing 'message')
    mock_glog_error = mocker.patch('memory_manager.glog.error')

    mem_manager.summarize_memory() # Explicitly call summarize_memory

    mock_ollama_chat.assert_called_once() # Should be called once
    mock_glog_error.assert_called_once() # Ensure error is logged
    assert mem_manager.get_summary() == "Failed to generate summary."

    # Test with missing 'content' in message
    mock_ollama_chat.reset_mock()
    mock_glog_error.reset_mock()
    mock_ollama_chat.return_value = {'message': {'unexpected_key': 'value'}} # Simulate malformed response (Missing 'content')

    mem_manager.summarize_memory() # Explicitly call summarize_memory again

    mock_ollama_chat.assert_called_once() # Should be called once again
    mock_glog_error.assert_called_once() # Ensure error is logged again
    assert mem_manager.get_summary() == "Failed to generate summary."