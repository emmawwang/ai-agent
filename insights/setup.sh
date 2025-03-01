#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data_cache logs data

echo "Setup complete!"
echo "Run 'python -m uvicorn app:app --reload' to start the application"
echo "To deactivate the virtual environment, run: deactivate" 