#!/bin/bash

# Feynomenon AI Setup and Management Script
# This script handles environment setup, dependency installation, and server management

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.8"
        
        if python3 -c "import sys; exit(0 if tuple(map(int, sys.version.split('.')[:2])) >= (3, 8) else 1)"; then
            print_success "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
            return 0
        else
            print_error "Python $PYTHON_VERSION found, but $REQUIRED_VERSION or higher is required"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        return 1
    fi
}

# Function to setup virtual environment
setup_virtual_env() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed successfully"
}

# Function to setup environment file
setup_env_file() {
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp env_example.txt .env
        print_warning "Please edit .env file and add your GEMINI_API_KEY"
        print_status "You can get your API key from: https://makersuite.google.com/app/apikey"
    else
        print_status ".env file already exists"
    fi
}

# Function to check API key
check_api_key() {
    if [ -f ".env" ]; then
        if grep -q "GEMINI_API_KEY=your_gemini_api_key_here" .env; then
            print_warning "Please update your GEMINI_API_KEY in the .env file"
            print_status "You can get your API key from: https://makersuite.google.com/app/apikey"
            return 1
        else
            print_success "API key appears to be configured"
            return 0
        fi
    else
        print_error ".env file not found. Run setup first."
        return 1
    fi
}

# Function to start the API server
start_server() {
    print_status "Starting Feynomenon AI API server..."
    print_status "Server will be available at: http://localhost:8000"
    print_status "API documentation: http://localhost:8000/docs"
    print_status "Press Ctrl+C to stop the server"
    echo ""
    
    # Activate virtual environment and start server
    source venv/bin/activate
    python api.py
}

# Function to start CLI interface
start_cli() {
    print_status "Starting Feynomenon AI CLI interface..."
    print_status "Press Ctrl+C to exit"
    echo ""
    
    # Activate virtual environment and start CLI
    source venv/bin/activate
    python cli.py
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    source venv/bin/activate
    
    if command_exists pytest; then
        pytest
    else
        print_warning "pytest not found. Installing..."
        pip install pytest
        pytest
    fi
}

# Function to show help
show_help() {
    echo "Feynomenon AI Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Initial setup (install dependencies, create .env)"
    echo "  start     - Start the API server"
    echo "  cli       - Start the command-line interface"
    echo "  test      - Run tests"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup    # First time setup"
    echo "  $0 start    # Start API server"
    echo "  $0 cli      # Start CLI interface"
}

# Main script logic
case "${1:-help}" in
    "setup")
        print_status "Setting up Feynomenon AI..."
        
        # Check Python version
        if ! check_python_version; then
            exit 1
        fi
        
        # Setup virtual environment
        setup_virtual_env
        
        # Install dependencies
        install_dependencies
        
        # Setup environment file
        setup_env_file
        
        print_success "Setup completed successfully!"
        echo ""
        print_status "Next steps:"
        echo "1. Edit .env file and add your GEMINI_API_KEY"
        echo "2. Run '$0 start' to start the API server"
        echo "3. Run '$0 cli' to start the command-line interface"
        ;;
    
    "start")
        # Check if setup is complete
        if [ ! -d "venv" ]; then
            print_error "Virtual environment not found. Run '$0 setup' first."
            exit 1
        fi
        
        if ! check_api_key; then
            print_error "Please configure your API key before starting the server."
            exit 1
        fi
        
        start_server
        ;;
    
    "cli")
        # Check if setup is complete
        if [ ! -d "venv" ]; then
            print_error "Virtual environment not found. Run '$0 setup' first."
            exit 1
        fi
        
        if ! check_api_key; then
            print_error "Please configure your API key before starting the CLI."
            exit 1
        fi
        
        start_cli
        ;;
    
    "test")
        # Check if setup is complete
        if [ ! -d "venv" ]; then
            print_error "Virtual environment not found. Run '$0 setup' first."
            exit 1
        fi
        
        run_tests
        ;;
    
    "help"|*)
        show_help
        ;;
esac 