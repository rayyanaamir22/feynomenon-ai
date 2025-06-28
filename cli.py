"""
Command-line interface for Feynomenon AI chat.
"""
from chat import Chat
import sys
from typing import NoReturn

def main() -> NoReturn:
    """Run the chat interface in command-line mode.
    
    This function handles the complete chat flow including:
    1. Topic gathering phase
    2. Feynman tutoring phase
    3. Error handling and graceful exit
    
    The function runs indefinitely until the user chooses to exit
    or an error occurs.
    """
    print("🤖 Welcome to Feynomenon AI - Your Feynman Technique Tutor!")
    print("=" * 60)
    
    try:
        # Initialize chat session
        session = Chat()
        
        # Start topic gathering phase
        print("\n--- Phase 1: Identifying Your Learning Topic ---")
        initial_message = session.start_topic_gathering()
        print(f"AI: {initial_message}")
        
        # Topic gathering loop - continue until topic is identified
        while session.current_phase == "topic_gathering":
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "stop"]:
                print("AI: Goodbye! Thanks for stopping by.")
                return
            
            # Process user message and get AI response
            result = session.process_topic_message(user_input)
            print(f"AI: {result['response']}")
            
            # Check if topic has been successfully identified
            if result["topic_identified"]:
                print(f"\nAI: Great! So, your chosen topic is: **{result['topic']}**")
                print("AI: Let's begin our Feynman technique learning journey.")
                break
        
        # Start Feynman tutoring phase
        print(f"\n--- Phase 2: Feynman Tutor for '{session.chosen_topic}' ---")
        feynman_response = session.start_feynman_tutoring()
        print(f"AI: {feynman_response}")
        
        # Feynman tutoring loop - continue until user exits
        while session.current_phase == "feynman_tutoring":
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "stop"]:
                print("AI: Thanks for learning with me today! Goodbye!")
                break
            
            # Process user message and get AI response
            result = session.process_feynman_message(user_input)
            print(f"AI: {result['response']}")
            
            # Check if session should end
            if result.get("session_ended"):
                break
    
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\nAI: Session interrupted. Goodbye!")
    except Exception as e:
        # Handle any other errors
        print(f"\n❌ An error occurred: {e}")
        print("Please ensure your GEMINI_API_KEY is correctly configured.")
        sys.exit(1)

if __name__ == "__main__":
    main() 