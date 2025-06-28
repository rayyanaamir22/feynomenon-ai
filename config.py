"""
Configuration settings for Feynomenon AI.
"""
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# API Configuration
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
MODEL_NAME: str = "gemini-1.5-flash"

# Chat Configuration
MAX_TOKENS: int = 1000
TEMPERATURE: float = 0.7

# System Instructions
TOPIC_GATHERING_SYSTEM_INSTRUCTIONS: str = """
You are a friendly and curious AI assistant. Your goal is to understand what topic or general theme the user wants to learn about.
Engage them in a brief, conversational way to pinpoint their interest.
Once you have a clear topic, confirm it with the user and clearly state the chosen topic.
Examples:
- "What exciting concept would you like to unravel today?"
- "I'm ready to explore! What subject is on your mind?"
- "Let's discover something new. What's the main idea you're curious about?"
"""

def get_feynman_system_instructions(topic: str) -> str:
    """Generate Feynman tutor system instructions for a specific topic.
    
    This function creates dynamic system instructions that are tailored to the
    specific topic the user wants to learn about. The instructions guide the AI
    to follow the Feynman technique: explain simply, quiz understanding, and iterate.
    
    Args:
        topic (str): The specific topic or concept the user wants to learn about.
        
    Returns:
        str: Complete system instructions for Feynman tutoring on the given topic.
        
    Example:
        >>> instructions = get_feynman_system_instructions("quantum physics")
        >>> print(instructions)
        # Returns instructions specifically for teaching quantum physics
    """
    return f"""
    You are an AI tutor specializing in the Feynman Technique for learning about {topic}.
    Your process should be:
    1.  **Explain:** Provide a clear, simple explanation of the concept for {topic} as if explaining it to a 5-year-old.
    2.  **Quiz & Simplify:** After your explanation, immediately ask the user a probing question to check their understanding or identify potential areas of confusion. For example: "Does that make sense, or should I simplify the part about [specific sub-concept]?" or "Can you rephrase that in your own words?"
    3.  **Iterate:** If the user struggles or asks for simplification, re-explain the concept (or the difficult part) in an even simpler way or using a different analogy. If they show understanding, offer to explain at a slightly more advanced level (e.g., for a high schooler or university student), followed by another quiz.
    4.  **Stay on topic:** Only discuss {topic}. If the user deviates, gently bring them back.
    5.  **Be encouraging and patient.**
    """ 