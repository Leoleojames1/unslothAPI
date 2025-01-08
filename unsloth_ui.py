# unsloth_ui.py
import gradio as gr
import requests
import json
import time
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import imageio
import os

# Configuration
API_URL = "http://localhost:8000"
SUPPORTED_MODELS = [
    # Llama 3.1 Models
    "Llama-3.1-8B",
    
    # Phi Models
    "Phi-3-medium",
    
    # Llama 3.2 Models
    "Llama-3.2-1B",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B",
    "Llama-3.2-3B-Instruct"
]

GGUF_METHODS = ["q8_0", "q4_k_m", "q5_k_m", "f16"]

def create_visualization(step, loss, save_path=None):
    """Create a 3D visualization of neural activity and optionally save as a video"""
    size = 5
    x, y, z = np.meshgrid(
        np.linspace(0, 1, size),
        np.linspace(0, 1, size),
        np.linspace(0, 1, size)
    )
    
    colors = np.sin(step / 10 + (x + y + z) * 2 * np.pi)
    intensity = np.exp(-loss) if loss else 1.0
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            mode='markers',
            marker=dict(
                size=10,
                color=colors.flatten(),
                colorscale='Rainbow',
                opacity=intensity
            )
        )
    ])
    
    fig.update_layout(
        title=f"Neural Network Activity (Step {step})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    if save_path:
        pio.write_image(fig, save_path)
    
    return fig

class UnslothUI:
    def __init__(self):
        self.current_model = None
        self.current_model_type = None
        self.training_status = "idle"
    
    def get_models(self):
        """Get list of available models from API"""
        try:
            response = requests.get(f"{API_URL}/models")
            models = response.json()
            return [model["name"] for model in models]
        except Exception as e:
            return []
    
    def start_training(self, model_name, dataset_name, batch_size, epochs, learning_rate):
        """Start model training via API"""
        try:
            config = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "num_epochs": epochs,
                "learning_rate": learning_rate
            }
            
            response = requests.post(f"{API_URL}/train", json=config)
            if response.status_code == 200:
                self.training_status = "training"
                return "Training started successfully"
            else:
                return f"Error starting training: {response.json()['detail']}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def check_training_status(self):
        """Check current training status and update visualization"""
        try:
            response = requests.get(f"{API_URL}/training_status")
            status = response.json()
            
            if status["status"] == "training":
                step = int(status["progress"] * 100)
                save_path = f"outputs/visualization_step_{step}.png"
                fig = create_visualization(
                    step=step,
                    loss=status["loss"],
                    save_path=save_path
                )
                return (
                    f"Training in progress: {status['progress']:.1f}% complete\n"
                    f"Current loss: {status['loss']:.4f}",
                    fig
                )
            else:
                self.training_status = status["status"]
                # Create video or GIF from saved images
                images = []
                for file_name in sorted(os.listdir("outputs")):
                    if file_name.startswith("visualization_step_") and file_name.endswith(".png"):
                        images.append(imageio.imread(os.path.join("outputs", file_name)))
                if images:
                    imageio.mimsave("outputs/visualization.gif", images, fps=10)
                    imageio.mimsave("outputs/visualization.mp4", images, fps=10)
                return f"Status: {status['status']}", None
                
        except Exception as e:
            return f"Error checking status: {str(e)}", None
    
    def convert_model(self, model_path, conversion_type, quant_method="q8_0"):
        """Convert model via API"""
        try:
            config = {
                "model_path": model_path,
                "conversion_type": conversion_type,
                "quantization_method": quant_method
            }
            
            response = requests.post(f"{API_URL}/convert", json=config)
            if response.status_code == 200:
                result = response.json()
                return f"Conversion successful. Model saved to: {result['path']}"
            else:
                return f"Error during conversion: {response.json()['detail']}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_ui(self):
        """Create the Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🤖 Unsloth Training Interface")
            
            with gr.Tabs():
                # Training Tab
                with gr.TabItem("Train"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## 🧠 Model Settings")
                            model_dropdown = gr.Dropdown(
                                choices=SUPPORTED_MODELS,
                                value="Llama-3.2-3B",
                                label="Select Model"
                            )
                            dataset_input = gr.Textbox(
                                label="Dataset Name",
                                value="Borcherding/OARC_Commander_v001"
                            )
                        
                        with gr.Column(scale=2):
                            visualization = gr.Plot(label="Neural Network Visualization")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## 📊 Training Settings")
                            batch_size = gr.Slider(1, 8, value=2, step=1, label="Batch Size")
                            epochs = gr.Slider(1, 10, value=1, step=1, label="Epochs")
                            learning_rate = gr.Number(value=2e-4, label="Learning Rate")
                            train_btn = gr.Button("Start Training")
                        
                        with gr.Column():
                            training_status = gr.Textbox(label="Training Status", interactive=False)
                
                # Model Management Tab
                with gr.TabItem("Model Management"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## 🔄 Model Operations")
                            models_dropdown = gr.Dropdown(
                                choices=self.get_models(),
                                label="Select Model",
                                interactive=True
                            )
                            refresh_btn = gr.Button("🔄 Refresh Models")
                            
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("### GGUF Conversion")
                                    quant_dropdown = gr.Dropdown(
                                        choices=GGUF_METHODS,
                                        value="q8_0",
                                        label="Quantization Method"
                                    )
                                    gguf_btn = gr.Button("Convert to GGUF")
                                
                                with gr.Column():
                                    gr.Markdown("### Model Merging")
                                    merge_btn = gr.Button("Merge Model")
                        
                        with gr.Column():
                            operation_status = gr.Textbox(
                                label="Operation Status",
                                interactive=False,
                                lines=4
                            )
            
            # Event handlers
            def update_models():
                models = self.get_models()
                return gr.Dropdown(choices=models)
            
            refresh_btn.click(
                fn=update_models,
                outputs=[models_dropdown]
            )
            
            train_btn.click(
                fn=self.start_training,
                inputs=[model_dropdown, dataset_input, batch_size, epochs, learning_rate],
                outputs=training_status
            )
            
            gguf_btn.click(
                fn=lambda m, q: self.convert_model(m, "gguf", q),
                inputs=[models_dropdown, quant_dropdown],
                outputs=operation_status
            )
            
            merge_btn.click(
                fn=lambda m: self.convert_model(m, "merge"),
                inputs=[models_dropdown],
                outputs=operation_status
            )
            
            # Training status update
            if self.training_status == "training":
                gr.Progress()
                
                def update_status():
                    while self.training_status == "training":
                        status, fig = self.check_training_status()
                        if fig is not None:
                            yield status, fig
                        time.sleep(1)
                
                gr.add_event_handler(
                    "load",
                    update_status,
                    outputs=[training_status, visualization]
                )
        
        return app

if __name__ == "__main__":
    ui = UnslothUI()
    app = ui.create_ui()
    app.launch(share=False)