"""
FastAPI application for Feynomenon AI chat interface.
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import uuid
from chat import Chat

app = FastAPI(title="Feynomenon AI", description="AI tutor using Feynman technique")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active chat sessions
active_sessions: Dict[str, Chat] = {}

class ChatRequest(BaseModel):
    """Request model for chat messages."""
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    session_id: str
    topic_identified: Optional[bool] = None
    topic: Optional[str] = None
    session_ended: Optional[bool] = None
    phase: str

class SessionState(BaseModel):
    """Model for session state information."""
    session_id: str
    current_phase: str
    chosen_topic: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint for processing messages.
    
    This endpoint handles both topic gathering and Feynman tutoring phases.
    It automatically creates new sessions and manages the conversation flow.
    
    Args:
        request (ChatRequest): The chat request containing user message and optional session ID.
        
    Returns:
        ChatResponse: The AI's response with session information.
        
    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        # Get or create session
        session_id = request.session_id
        if not session_id or session_id not in active_sessions:
            # Create new session with unique ID
            session_id = str(uuid.uuid4())
            active_sessions[session_id] = Chat()
            # Start topic gathering phase
            initial_message = active_sessions[session_id].start_topic_gathering()
            return ChatResponse(
                response=initial_message,
                session_id=session_id,
                topic_identified=False,
                phase="topic_gathering"
            )
        
        session = active_sessions[session_id]
        
        # Process message based on current phase
        if session.current_phase == "topic_gathering":
            result = session.process_topic_message(request.message)
            
            if result["topic_identified"]:
                # Topic identified - transition to Feynman tutoring
                feynman_response = session.start_feynman_tutoring()
                return ChatResponse(
                    response=feynman_response,
                    session_id=session_id,
                    topic_identified=True,
                    topic=result["topic"],
                    phase="feynman_tutoring"
                )
            else:
                # Continue topic gathering
                return ChatResponse(
                    response=result["response"],
                    session_id=session_id,
                    topic_identified=False,
                    phase="topic_gathering"
                )
        
        elif session.current_phase == "feynman_tutoring":
            result = session.process_feynman_message(request.message)
            
            # Clean up session if user wants to end
            if result.get("session_ended"):
                del active_sessions[session_id]
            
            return ChatResponse(
                response=result["response"],
                session_id=session_id,
                session_ended=result.get("session_ended", False),
                phase=session.current_phase
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/state", response_model=SessionState)
async def get_session_state(session_id: str) -> SessionState:
    """Get the current state of a chat session.
    
    Args:
        session_id (str): The unique identifier for the session.
        
    Returns:
        SessionState: Current session state information.
        
    Raises:
        HTTPException: If session is not found.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    state = session.get_session_state()
    
    return SessionState(
        session_id=session_id,
        current_phase=state["current_phase"],
        chosen_topic=state["chosen_topic"]
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a chat session.
    
    Args:
        session_id (str): The unique identifier for the session to delete.
        
    Returns:
        Dict[str, str]: Confirmation message.
        
    Raises:
        HTTPException: If session is not found.
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint.
    
    Returns:
        Dict[str, Any]: Health status and active session count.
    """
    return {"status": "healthy", "active_sessions": len(active_sessions)}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time chat communication.
    
    This endpoint provides real-time bidirectional communication for chat.
    It automatically manages session creation and cleanup.
    
    Args:
        websocket (WebSocket): The WebSocket connection.
        session_id (str): The unique identifier for the session.
    """
    await websocket.accept()
    
    try:
        # Get or create session
        if session_id not in active_sessions:
            active_sessions[session_id] = Chat()
            # Send initial message for topic gathering
            initial_message = active_sessions[session_id].start_topic_gathering()
            await websocket.send_text(json.dumps({
                "type": "message",
                "response": initial_message,
                "phase": "topic_gathering",
                "topic_identified": False
            }))
        
        session = active_sessions[session_id]
        
        # Main WebSocket message loop
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            # Process message based on current phase
            if session.current_phase == "topic_gathering":
                result = session.process_topic_message(user_message)
                
                if result["topic_identified"]:
                    # Topic identified - start Feynman tutoring
                    feynman_response = session.start_feynman_tutoring()
                    await websocket.send_text(json.dumps({
                        "type": "message",
                        "response": feynman_response,
                        "phase": "feynman_tutoring",
                        "topic_identified": True,
                        "topic": result["topic"]
                    }))
                else:
                    # Continue topic gathering
                    await websocket.send_text(json.dumps({
                        "type": "message",
                        "response": result["response"],
                        "phase": "topic_gathering",
                        "topic_identified": False
                    }))
            
            elif session.current_phase == "feynman_tutoring":
                result = session.process_feynman_message(user_message)
                
                # Clean up session if user wants to end
                if result.get("session_ended"):
                    del active_sessions[session_id]
                
                await websocket.send_text(json.dumps({
                    "type": "message",
                    "response": result["response"],
                    "phase": session.current_phase,
                    "session_ended": result.get("session_ended", False)
                }))
    
    except WebSocketDisconnect:
        # Clean up session on disconnect
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        # Send error message to client
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 