"""
Chat management for Feynomenon AI.
"""
import google.generativeai as genai
from typing import Optional, Dict, Any
from config import (
    GEMINI_API_KEY, 
    MODEL_NAME, 
    TOPIC_GATHERING_SYSTEM_INSTRUCTIONS,
    get_feynman_system_instructions
)

class Chat:
    """Manages a chat session with topic gathering and Feynman tutoring phases."""
    
    def __init__(self) -> None:
        """Initialize the chat session.
        
        Raises:
            ValueError: If GEMINI_API_KEY is not found in environment variables.
        """
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.topic_gathering_model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=TOPIC_GATHERING_SYSTEM_INSTRUCTIONS
        )
        self.topic_chat = None
        self.feynman_chat = None
        self.chosen_topic = None
        self.current_phase = "topic_gathering"
        
    def start_topic_gathering(self) -> str:
        """Start the topic gathering phase and return initial message.
        
        Returns:
            str: The initial greeting message for topic gathering.
        """
        self.topic_chat = self.topic_gathering_model.start_chat(history=[])
        return "Hello! I'm here to help you learn. What topic or concept are you curious about today?"
    
    def process_topic_message(self, user_input: str) -> Dict[str, Any]:
        """Process a message during the topic gathering phase.
        
        Args:
            user_input (str): The user's message to process.
            
        Returns:
            Dict[str, Any]: Response containing AI reply and topic identification status.
                Keys:
                - response (str): AI's response message
                - topic_identified (bool): Whether a topic was successfully identified
                - topic (Optional[str]): The identified topic (if any)
                
        Raises:
            RuntimeError: If topic gathering chat is not initialized.
        """
        if not self.topic_chat:
            raise RuntimeError("Topic gathering chat not initialized")
        
        # Send user message to Gemini and get response
        response = self.topic_chat.send_message(user_input)
        
        # Check if topic has been identified from the AI's response
        topic_identified = self._extract_topic_from_response(response.text)
        
        if topic_identified:
            # Topic found - transition to Feynman tutoring phase
            self.chosen_topic = topic_identified
            self.current_phase = "feynman_tutoring"
            return {
                "response": f"Great! So, your chosen topic is: **{topic_identified.title()}**. Let's begin our Feynman technique learning journey.",
                "topic_identified": True,
                "topic": topic_identified.title()
            }
        
        # Topic not yet identified - continue gathering
        return {
            "response": response.text,
            "topic_identified": False,
            "topic": None
        }
    
    def start_feynman_tutoring(self) -> str:
        """Start the Feynman tutoring phase and return initial message.
        
        Returns:
            str: The initial Feynman explanation message.
            
        Raises:
            RuntimeError: If no topic has been identified yet.
        """
        if not self.chosen_topic:
            raise RuntimeError("No topic has been identified yet")
        
        # Generate topic-specific system instructions for Feynman technique
        feynman_system_instructions = get_feynman_system_instructions(self.chosen_topic)
        feynman_model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=feynman_system_instructions
        )
        self.feynman_chat = feynman_model.start_chat(history=[])
        
        # Send initial prompt to start the Feynman explanation
        first_prompt = f"Please begin explaining {self.chosen_topic} using the Feynman technique."
        response = self.feynman_chat.send_message(first_prompt)
        return response.text
    
    def process_feynman_message(self, user_input: str) -> Dict[str, Any]:
        """Process a message during the Feynman tutoring phase.
        
        Args:
            user_input (str): The user's message to process.
            
        Returns:
            Dict[str, Any]: Response containing AI reply and session status.
                Keys:
                - response (str): AI's response message
                - session_ended (bool): Whether the session should be terminated
                
        Raises:
            RuntimeError: If Feynman tutoring chat is not initialized.
        """
        if not self.feynman_chat:
            raise RuntimeError("Feynman tutoring chat not initialized")
        
        # Check for session termination commands
        if user_input.lower() in ["quit", "exit", "stop"]:
            return {
                "response": "Thanks for learning with me today! Goodbye!",
                "session_ended": True
            }
        
        # Send user message to Gemini and get response
        response = self.feynman_chat.send_message(user_input)
        return {
            "response": response.text,
            "session_ended": False
        }
    
    def _extract_topic_from_response(self, response_text: str) -> Optional[str]:
        """Extract the identified topic from the AI's response.
        
        This method uses pattern matching to identify when the AI has confirmed
        a topic. It looks for phrases like "topic is" or "chosen topic is"
        and extracts the topic name that follows.
        
        Args:
            response_text (str): The AI's response text to analyze.
            
        Returns:
            Optional[str]: The extracted topic name, or None if no topic found.
        """
        response_lower = response_text.lower()
        
        # Look for topic confirmation patterns in the AI's response
        if "topic is" in response_lower or "chosen topic is" in response_lower:
            try:
                # Split on "topic is" and take the part after it
                parts = response_lower.split("topic is")
                if len(parts) > 1:
                    topic_text = parts[1].strip()
                    # Remove common concluding phrases that might follow the topic
                    if "!" in topic_text: 
                        topic_text = topic_text.split("!")[0]
                    if "." in topic_text: 
                        topic_text = topic_text.split(".")[0]
                    return topic_text.strip()
            except Exception:
                # If extraction fails, return None to continue gathering
                pass
        
        return None
    
    def get_session_state(self) -> Dict[str, Any]:
        """Get the current state of the chat session.
        
        Returns:
            Dict[str, Any]: Current session state information.
                Keys:
                - current_phase (str): Current phase ("topic_gathering" or "feynman_tutoring")
                - chosen_topic (Optional[str]): The identified topic (if any)
                - topic_chat_initialized (bool): Whether topic gathering chat is ready
                - feynman_chat_initialized (bool): Whether Feynman tutoring chat is ready
        """
        return {
            "current_phase": self.current_phase,
            "chosen_topic": self.chosen_topic,
            "topic_chat_initialized": self.topic_chat is not None,
            "feynman_chat_initialized": self.feynman_chat is not None
        } 