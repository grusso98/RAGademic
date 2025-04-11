import json
import os
import glog
import ollama 

_MEMORY_FILE = "conversation_memory.json"
_SUMMARY_PROMPT = """
Based on the following conversation history between a User and an AI Tutor, 
provide a concise summary of the key topics discussed, decisions made, 
or important information shared. This summary will be used to give the AI 
context about past interactions.

Conversation History:
{history_text}

Concise Summary:
"""

class MemoryManager:
    """Manages long-term conversation memory, including saving, loading, and summarization."""

    def __init__(self, memory_file: str = _MEMORY_FILE, model_identifier: str = "llama3.2:latest"):
        """
        Initializes the MemoryManager.

        Args:
            memory_file (str): Path to the file for storing conversation history.
            model_identifier (str): The identifier for the LLM used for summarization (e.g., "llama3.2:latest", "gemma3:4b").
                                     Adjust if using OpenAI models.
        """
        self.memory_file = memory_file
        self.history = []  # List of [user_query, ai_response] pairs
        self.summary = ""
        self.model_identifier = model_identifier # Store model used for summarization
        self.load_memory()

    def load_memory(self) -> None:
        """Loads conversation history from the memory file and generates an initial summary."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                glog.info(f"Loaded {len(self.history)} interactions from {self.memory_file}")
                # Summarize the loaded history on startup
                self.summarize_memory() 
            else:
                glog.info(f"Memory file {self.memory_file} not found. Starting fresh.")
                self.history = []
                self.summary = "No previous conversation history found."
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            glog.error(f"Error loading memory file {self.memory_file}: {e}. Starting fresh.")
            self.history = []
            self.summary = "Error loading conversation history." # Indicate error in summary

    def save_memory(self) -> None:
        """Saves the current conversation history to the memory file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=4, ensure_ascii=False)
            # glog.debug(f"Saved {len(self.history)} interactions to {self.memory_file}") # Optional: Log successful save
        except IOError as e:
            glog.error(f"Error saving memory file {self.memory_file}: {e}")

    def add_interaction(self, user_query: str, ai_response: str) -> None:
        """
        Adds a user query and AI response pair to the history and saves it.

        Args:
            user_query (str): The user's input.
            ai_response (str): The AI's response.
        """
        self.history.append([user_query, ai_response])
        self.save_memory()
        # Optional: Re-summarize periodically (e.g., every 5 interactions)
        if len(self.history) % 5 == 0:
            self.summarize_memory()

    def get_history(self) -> list:
        """Returns the full conversation history."""
        return self.history

    def get_summary(self) -> str:
        """Returns the current conversation summary."""
        return self.summary

    def clear_memory(self) -> None:
        """Clears the conversation history and summary, and saves the empty state."""
        glog.info("Clearing conversation memory.")
        self.history = []
        self.summary = "Conversation history has been cleared."
        self.save_memory() 
        
    def summarize_memory(self) -> None:
        """
        Summarizes the conversation history using an LLM.
        Updates self.summary.
        """
        if not self.history:
            self.summary = "No conversation history to summarize."
            glog.info("No history to summarize.")
            return

        glog.info(f"Attempting to summarize {len(self.history)} interactions...")
        
        history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in self.history])
        
        # Limit history length if it's extremely long to avoid context window issues
        max_history_chars = 10000 
        if len(history_text) > max_history_chars:
            glog.warning(f"History text exceeds {max_history_chars} chars, truncating for summary prompt.")
            history_text = history_text[-max_history_chars:] 

        prompt = _SUMMARY_PROMPT.format(history_text=history_text)

        try:
            # --- Using Ollama for Summarization ---
            # Ensure self.model_identifier is set correctly during init
            # e.g., "llama3.2:latest" or similar depending on your Ollama setup
            response = ollama.chat(
                model=self.model_identifier, 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                    {"role": "user", "content": prompt}
                ],
                stream=False 
            )
            if response and 'message' in response and 'content' in response['message']:
                self.summary = response['message']['content'].strip()
                glog.info(f"Successfully generated memory summary using {self.model_identifier}.")
            else:
                glog.error("Failed to get valid summary content from Ollama response.")
                self.summary = "Failed to generate summary."

        except Exception as e:
            glog.error(f"Error during conversation summarization with {self.model_identifier}: {e}")
            self.summary = "Error occurred during summarization."


if __name__ == "__main__":
    # Example Usage (for testing the module directly)
    glog.setLevel("INFO") 
    
    # Specify the model you want to use for summarization here
    test_model = "llama3.2:latest" # Or "gemma3:4b", etc.
    
    mem_manager = MemoryManager(model_identifier=test_model)
    
    print("Initial Summary:", mem_manager.get_summary())
    print("Initial History:", mem_manager.get_history())

    if not mem_manager.get_history(): # Only add if history is empty
        print("\nAdding interactions...")
        mem_manager.add_interaction("What is the capital of France?", "The capital of France is Paris.")
        mem_manager.add_interaction("Can you recommend a book about AI?", "Sure, 'Superintelligence' by Nick Bostrom is a popular one.")
        print("Updated History:", mem_manager.get_history())
        
        # Manually trigger summarization after adding (since it only runs on load by default)
        print("\nManually triggering summarization...")
        mem_manager.summarize_memory()
        print("Updated Summary:", mem_manager.get_summary())
