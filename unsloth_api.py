import os
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from datetime import datetime
import torch
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path("models")
DATASET_DIR = Path("datasets")
OUTPUT_DIR = Path("outputs")
HF_TOKEN = os.getenv("HF_TOKEN")

# Ensure directories exist
for dir_path in [BASE_DIR, DATASET_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Unsloth Training Server")

# Model configurations
UNSLOTH_MODELS = {
    # Llama 3.1 Models
    "Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    
    # Phi Models
    "Phi-3-medium": "unsloth/Phi-3-medium-4k-instruct",
    
    # Llama 3.2 Models
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B-bnb-4bit",
    "Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "Llama-3.2-3B": "unsloth/Llama-3.2-3B-bnb-4bit",
    "Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
}

# Pydantic models for requests
class TrainingConfig(BaseModel):
    model_name: str
    dataset_name: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    
    model_config = {
        'protected_namespaces': ()
    }

class ModelConversionConfig(BaseModel):
    model_path: str
    conversion_type: str  # 'gguf' or 'merge'
    quantization_method: str = "q8_0"  # For GGUF conversion
    
    model_config = {
        'protected_namespaces': ()
    }

# Global state management
class TrainingState:
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_status = "idle"
        self.training_progress = 0.0
        self.current_loss = 0.0

training_state = TrainingState()

def setup_model(model_name: str):
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=UNSLOTH_MODELS[model_name],
            max_seq_length=2048,
            load_in_4bit=True,
            token=HF_TOKEN
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407
        )
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1"
        )
        
        # Ensure the model returns a loss
        model.config.return_dict = True
        model.config.output_hidden_states = False
        model.config.output_attentions = False
        
        return model, tokenizer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up model: {str(e)}")

def format_dataset(examples, tokenizer):
    conversations = [
        [
            {"role": "user", "content": inst},
            {"role": "assistant", "content": out}
        ]
        for inst, out in zip(examples['instruction'], examples['output'])
    ]
    
    formatted_data = []
    for i, conv in enumerate(conversations):
        result = tokenizer.apply_chat_template(conv, tokenize=True, add_generation_prompt=False)
        
        # Limit debug output to the first 5 results
        if i < 5:
            print(f"Conversation: {conv}")
            print(f"Result: {result}")
        
        # Check if result is a list or dictionary
        if isinstance(result, list):
            if i < 5:
                print("Result is a list")
            formatted_data.append({
                "input_ids": result,
                "labels": result
            })
        elif isinstance(result, dict):
            if i < 5:
                print("Result is a dictionary")
            if "input_ids" in result:
                formatted_data.append({
                    "input_ids": result["input_ids"],
                    "labels": result["input_ids"]
                })
        else:
            if i < 5:
                print(f"Unexpected result format: {result}")
    
    return {"input_ids": [data["input_ids"] for data in formatted_data], "labels": [data["labels"] for data in formatted_data]}

async def train_model(config: TrainingConfig):
    try:
        training_state.current_status = "setting_up"
        training_state.training_progress = 0.0
        
        # Setup model
        model, tokenizer = setup_model(config.model_name)
        training_state.current_model = model
        training_state.current_tokenizer = tokenizer
        
        # Load and process dataset
        training_state.current_status = "processing_dataset"
        dataset = load_dataset(config.dataset_name, split="train")
        
        # Format dataset
        dataset = dataset.map(
            lambda examples: format_dataset(examples, tokenizer),
            batched=True,
            batch_size=100,
        )
        
        # Setup training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / timestamp),
            per_device_train_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
        )
        
        # Configure trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="input_ids",
            max_seq_length=2048,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            args=training_args
        )
        
        # Start training
        training_state.current_status = "training"
        trainer.train()
        
        # Save model
        training_state.current_status = "saving"
        model_path = OUTPUT_DIR / timestamp
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        training_state.current_status = "completed"
        return {"status": "success", "model_path": str(model_path)}
        
    except Exception as e:
        training_state.current_status = "failed"
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    if training_state.current_status != "idle":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    background_tasks.add_task(train_model, config)
    return {"status": "Training started"}

@app.get("/training_status")
async def get_training_status():
    return {
        "status": training_state.current_status,
        "progress": training_state.training_progress,
        "loss": training_state.current_loss
    }

@app.post("/convert")
async def convert_model(config: ModelConversionConfig):
    try:
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
            
        if config.conversion_type == "gguf":
            # Load model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=UNSLOTH_MODELS["Llama-3.2-3B"],
                max_seq_length=2048,
                load_in_4bit=True,
            )
            
            # Load adapter if exists
            if (model_path / "adapter_model.safetensors").exists():
                model.load_adapter(str(model_path))
            
            # Convert to GGUF
            gguf_path = model_path.parent / f"{model_path.name}_gguf"
            model.save_pretrained_gguf(
                str(gguf_path),
                tokenizer,
                quantization_method=config.quantization_method
            )
            return {"status": "success", "path": str(gguf_path)}
            
        elif config.conversion_type == "merge":
            # Load and merge model
            merge_path = model_path.parent / f"{model_path.name}_merged"
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=UNSLOTH_MODELS["Llama-3.2-3B"],
                max_seq_length=2048,
                load_in_4bit=True,
            )
            model.load_adapter(str(model_path))
            model.save_pretrained_merged(str(merge_path), tokenizer, save_method="merged_16bit")
            return {"status": "success", "path": str(merge_path)}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List all available trained models"""
    try:
        models = []
        for path in OUTPUT_DIR.iterdir():
            if path.is_dir():
                model_type = "unknown"
                if (path / "adapter_model.safetensors").exists():
                    model_type = "adapter"
                elif (path / "model.safetensors").exists():
                    model_type = "merged"
                elif list(path.glob("*.gguf")):
                    model_type = "gguf"
                    
                models.append({
                    "name": path.name,
                    "type": model_type,
                    "path": str(path)
                })
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

if __name__ == "__main__":
    # Login to HuggingFace
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)