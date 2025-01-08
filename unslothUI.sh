# unslothUI.sh
#!/bin/bash

echo "Starting Unsloth Gradio UI..."

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate unsloth_env

# Start the UI
echo "Starting Gradio UI..."
python unsloth_ui.py

echo "UI should be running at http://localhost:7860"