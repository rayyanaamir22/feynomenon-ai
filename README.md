# Feynomenon AI

An AI tutor that uses the Feynman Technique to help users learn complex topics through simple explanations and interactive quizzing.

## Features

- **Two-Phase Learning Process**: 
  1. **Topic Gathering**: Identifies what the user wants to learn
  2. **Feynman Tutoring**: Explains concepts simply and quizzes understanding
- **Interactive Chat Interface**: Both CLI and API endpoints
- **Real-time Communication**: WebSocket support for frontend integration
- **Session Management**: Maintains conversation context across phases

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd feynomenon-ai

# Make the setup script executable (if not already)
chmod +x start.sh

# Run the setup script
./start.sh setup
```

The setup script will:
- Check Python version (requires 3.8+)
- Create a virtual environment
- Install all dependencies
- Create a `.env` file from template

### 2. Get API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Edit the `.env` file and add your API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### 3. Run the Application

#### Command Line Interface
```bash
./start.sh cli
```

#### API Server
```bash
./start.sh start
```
The API will be available at `http://localhost:8000`

#### Other Commands
```bash
./start.sh test    # Run tests
./start.sh help    # Show all available commands
```

## API Endpoints

### REST API

- `POST /chat` - Send a message and get AI response
- `GET /session/{session_id}/state` - Get session state
- `DELETE /session/{session_id}` - Delete a session
- `GET /health` - Health check

### WebSocket

- `WS /ws/{session_id}` - Real-time chat communication

## Usage Examples

### REST API Example

```python
import requests

# Start a new chat session
response = requests.post("http://localhost:8000/chat", json={
    "message": "I want to learn about quantum physics"
})

print(response.json())
# {
#   "response": "Hello! I'm here to help you learn...",
#   "session_id": "uuid-here",
#   "topic_identified": false,
#   "phase": "topic_gathering"
# }
```

### WebSocket Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session-id');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('AI:', data.response);
};

ws.send(JSON.stringify({
    message: "I want to learn about machine learning"
}));
```

## Project Structure

```
feynomenon-ai/
├── api.py              # FastAPI application
├── chat.py             # Chat session management
├── cli.py              # Command-line interface
├── config.py           # Configuration settings
├── start.sh            # Setup and management script
├── requirements.txt    # Python dependencies
├── env_example.txt     # Environment variables template
├── README.md          # This file
└── training_scripts/  # Fine-tuning scripts
    ├── train_gemini.py
    └── train_mistral.py
```

## Development

### Running Tests
```bash
./start.sh test
```

### API Documentation
When the API server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Manual Setup (Alternative)
If you prefer not to use the setup script:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp env_example.txt .env
# Edit .env and add your GEMINI_API_KEY

# Run application
python api.py    # For API server
python cli.py    # For CLI interface
```

## Troubleshooting

### Common Issues

1. **"Virtual environment not found"**
   - Run `./start.sh setup` first

2. **"API key not configured"**
   - Edit `.env` file and add your `GEMINI_API_KEY`

3. **"Python 3 not found"**
   - Install Python 3.8 or higher

4. **Permission denied on start.sh**
   - Run `chmod +x start.sh`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]
