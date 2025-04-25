import pytest
from unittest.mock import MagicMock, patch

from agent import is_context_sufficient, perform_web_search, scrape_web_content

# Fixture to mock ollama for is_context_sufficient tests
@pytest.fixture
def mock_ollama(mocker):
    mock_chat = mocker.patch('agent.ollama.chat')
    return mock_chat

# Fixture to mock DuckDuckGoSearchResults for perform_web_search tests
@pytest.fixture
def mock_ddg_search(mocker):
    mock_tool = MagicMock()
    mocker.patch('agent.DuckDuckGoSearchResults', return_value=mock_tool)
    return mock_tool

# Fixture to mock WebBaseLoader for scraping tests
@pytest.fixture
def mock_web_base_loader(mocker):
    mock_loader_instance = MagicMock()
    mocker.patch('agent.WebBaseLoader', return_value=mock_loader_instance)
    return mock_loader_instance

# Fixture to mock semantic_chunk_text and add_document for search_scrape_and_ingest tests
@pytest.fixture
def mock_ingestion_pipeline(mocker):
    mock_chunk_text = mocker.patch('agent.semantic_chunk_text')
    mock_add_document = mocker.patch('agent.add_document')
    return mock_chunk_text, mock_add_document


# Test cases for is_context_sufficient

def test_is_context_sufficient_relevant_refined(mock_ollama):
    """Tests when the checker model returns 'Relevant' with sufficient context."""
    mock_ollama.return_value = {
        'message': {'content': 'Answer: Relevant'}
    }
    query = "What is the capital of France?"
    # Use a context string that is at least 50 characters long
    context = "Paris is the capital of France. It is a major European city and a global center for art, fashion, gastronomy, and culture. Its picturesque 19th-century cityscape is crisscrossed by wide boulevards and the River Seine."

    # Call the function
    result = is_context_sufficient(query, context)

    # Assert that the mock was called exactly once
    mock_ollama.assert_called_once()

    # Then assert the return value
    assert result is True

def test_is_context_sufficient_insufficient(mock_ollama):
    """Tests when the checker model returns 'Insufficient' with sufficient context."""
    mock_ollama.return_value = {
        'message': {'content': 'Answer: Insufficient'}
    }
    query = "What is the capital of France?"
    # Use a context string that is at least 50 characters long but irrelevant
    context = "The weather today is sunny with a high of 75 degrees Fahrenheit. There will be a slight breeze in the afternoon. Don't forget to bring sunscreen if you plan to be outdoors for an extended period."
    assert is_context_sufficient(query, context) is False
    mock_ollama.assert_called_once()

def test_is_context_sufficient_empty_context():
    """Tests with empty context."""
    query = "Test query"
    context = ""
    assert is_context_sufficient(query, context) is False

def test_is_context_sufficient_short_context():
    """Tests with very short context."""
    query = "Test query"
    context = "Short."
    assert is_context_sufficient(query, context) is False

def test_is_context_sufficient_retrieval_failure_context():
    """Tests with context indicating retrieval failure."""
    query = "Test query"
    context = "No relevant documents found."
    # Even though these are short, they are specific error strings handled before the length check
    assert is_context_sufficient(query, context) is False

    context = "Error retrieving context from documents."
    assert is_context_sufficient(query, context) is False


def test_is_context_sufficient_ollama_error(mock_ollama):
    """Tests when ollama.chat raises an exception with sufficient context."""
    mock_ollama.side_effect = Exception("Ollama connection error")
    query = "Test query"
    # Use a context string that is at least 50 characters long
    context = "This is a context string that is long enough to pass the initial length check and should trigger the ollama call."
    assert is_context_sufficient(query, context) is False
    mock_ollama.assert_called_once()

def test_is_context_sufficient_malformed_ollama_response(mock_ollama):
    """Tests with a malformed response from ollama with sufficient context."""
    query = "This is a test query." # Added: Define the query variable
    # Use a context string that is at least 50 characters long
    context = "This is another context string that is long enough to pass the initial length check."

    mock_ollama.return_value = {'unexpected_key': 'value'} # Missing 'message'
    assert is_context_sufficient(query, context) is False
    mock_ollama.assert_called_once() # Check if called for the first assertion

    mock_ollama.reset_mock() # Reset the mock to check the next call
    mock_ollama.return_value = {'message': {'unexpected_key': 'value'}} # Missing 'content'
    assert is_context_sufficient(query, context) is False
    mock_ollama.assert_called_once() # Check if called for the second assertion


def test_perform_web_search_success(mock_ddg_search):
    """Tests successful web search and parsing."""
    mock_ddg_search.run.return_value = (
        "snippet: snippet 1, title: Title 1, link: http://example.com/page1\n"
        "snippet: snippet 2, title: Title 2, link: https://anothersite.org/page2"
    )
    query = "test search"
    results = perform_web_search(query, num_results=2)

    mock_ddg_search.run.assert_called_once_with(query)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]['title'] == "Title 1"
    assert results[0]['link'] == "http://example.com/page1"
    assert results[0]['snippet'] == "snippet 1"
    assert results[1]['title'] == "Title 2"
    assert results[1]['link'] == "https://anothersite.org/page2"
    assert results[1]['snippet'] == "snippet 2"

def test_perform_web_search_no_results_from_ddg(mock_ddg_search):
    """Tests when DuckDuckGo returns no results."""
    mock_ddg_search.run.return_value = "No good DuckDuckGo Search Result was found"
    query = "empty search"
    results = perform_web_search(query)

    mock_ddg_search.run.assert_called_once_with(query)
    assert results == []

def test_perform_web_search_parsing_failure(mock_ddg_search):
    """Tests when parsing the DDG output fails."""
    mock_ddg_search.run.return_value = "Malformed search result string without expected pattern."
    query = "parse failure"
    results = perform_web_search(query)

    mock_ddg_search.run.assert_called_once_with(query)
    assert results == []

def test_perform_web_search_error(mock_ddg_search):
    """Tests when DuckDuckGo search raises an exception."""
    mock_ddg_search.run.side_effect = Exception("DDG API error")
    query = "error search"
    results = perform_web_search(query)

    mock_ddg_search.run.assert_called_once_with(query)
    assert results == []

def test_perform_web_search_invalid_link_in_result(mock_ddg_search):
    """Tests skipping results with invalid links."""
    mock_ddg_search.run.return_value = (
        "snippet: valid snippet, title: Valid Title, link: http://goodlink.com\n"
        "snippet: invalid snippet, title: Invalid Title, link: invalid-link" # Invalid link
    )
    query = "invalid link search"
    results = perform_web_search(query)

    assert len(results) == 1
    assert results[0]['link'] == "http://goodlink.com"

def test_scrape_web_content_success(mock_web_base_loader):
    """Tests successful web scraping."""
    mock_document = MagicMock()
    mock_document.page_content = "This is the scraped content."
    mock_web_base_loader.load.return_value = [mock_document]

    url = "http://testurl.com"
    content = scrape_web_content(url)

    mock_web_base_loader.load.assert_called_once_with()
    assert content == "This is the scraped content."

def test_scrape_web_content_no_documents(mock_web_base_loader):
    """Tests when WebBaseLoader returns no documents."""
    mock_web_base_loader.load.return_value = []

    url = "http://nodocuments.com"
    content = scrape_web_content(url)

    mock_web_base_loader.load.assert_called_once_with()
    assert content is None

def test_scrape_web_content_empty_content(mock_web_base_loader):
    """Tests when WebBaseLoader returns documents with empty content."""
    mock_document = MagicMock()
    mock_document.page_