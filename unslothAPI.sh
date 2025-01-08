# unslothAPI.sh
#!/bin/bash

echo "Starting Unsloth FastAPI Server..."

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate unsloth_env

# Start the server
echo "Starting FastAPI server..."
uvicorn unsloth_api:app --host 0.0.0.0 --port 8000 --reload

echo "Server should be running at http://localhost:8000"
echo "Press Ctrl+C to stop the server"