import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration Constants
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./fine_tuned_weights"


def prepare_mock_dataset():
    """
    Creates a small instructional dataset for demonstration.
    In a real scenario, you would load a JSON/CSV file with thousands of lines.
    """
    # The format follows standard LLM chat instruction styles (System, User, Assistant)
    training_data = {
        "text": [
            "<|im_start|>system\nYou are a legal assistant.<|im_end|>\n<|im_start|>user\nWhat is the late fee policy?<|im_end|>\n<|im_start|>assistant\nThe late fee policy dictates an immediate $50 penalty plus 1.5% monthly compounding interest on outstanding invoice balances.<|im_end|>",
            "<|im_start|>system\nYou are a legal assistant.<|im_end|>\n<|im_start|>user\nWho owns the intellectual property?<|im_end|>\n<|im_start|>assistant\nAll source code, design blueprints, and documentation products created remain 100% owned by the primary developer.<|im_end|>"
        ]
    }
    return Dataset.from_dict(training_data)


def run_qlora_training():
    """
    Executes a high-efficiency QLoRA fine-tuning sequence using PyTorch.
    Optimized to fit within modest GPU systems (like free Google Colab/Kaggle instances).
    """
    print("⏳ Starting QLoRA Fine-Tuning Pipeline Setup...")
    
    # 1. Hardware Check - QLoRA requires a CUDA compatible NVIDIA GPU to run bitsandbytes
    if not torch.cuda.is_available():
        print("⚠️ NVIDIA GPU (CUDA) not found locally.")
        print("💡 ML Tip: Copy this script onto a free Google Colab or Kaggle notebook instance with a T4 GPU to see it train live!")
        return

    # 2. BitsAndBytes 4-bit Quantization Configuration
    # Drops VRAM consumption drastically by packing 16-bit float weights into 4-bit integers
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # NormalFloat4: Optimal math distribution for neural weights
        bnb_4bit_compute_dtype=torch.float16,  # Compute matrix additions in half-precision float16
        bnb_4bit_use_double_quant=True         # Compresses quantization constants to save extra memory
    )

    # 3. Load Base Model and Tokenizer
    print(f"📦 Downloading base model structure: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Map padding tokens to stop tokens to prevent layout drift

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 4. Prepare Model Weights for K-Bit Low Precision Training
    base_model = prepare_model_for_kbit_training(base_model)

    # 5. PEFT / LoRA Adapter Layer Topologies Config
    # Instead of modifying 1.5 billion parameters, we freeze the model and train tiny rank matrix layers
    lora_config = LoraConfig(
        r=8,                       # Rank matrix size (lower value = less memory, higher = smarter adaptability)
        lora_alpha=16,             # Mathematical scaling factor weight multiplier
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Inject LoRA layers inside Attention nodes
        lora_dropout=0.05,         # Prevents model from overfitting on limited data
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Attach the tiny custom weight matrix layers to our frozen model framework
    model = get_peft_model(base_model, lora_config)
    print("✅ Model successfully wrapped with LoRA trainable layers.")
    model.print_trainable_parameters() # Prints exact ratio of trainable vs frozen weights

    # 6. Define Hyper-parameters and Training Context Configurations
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,     # Small batch size to ensure zero Out-Of-Memory (OOM) errors
        gradient_accumulation_steps=4,     # Simulates a larger batch size mathematically
        warmup_steps=2,
        max_steps=10,                      # Low steps count for demonstration speed
        learning_rate=2e-4,                # Ideal standard learning rate scalar for adapter tuning
        fp16=True,                         # Half-precision training accelerator loop
        logging_steps=1,
        save_strategy="no",
        report_to="none"                   # Change to "wandb" if utilizing Weights & Biases telemetry tracking
    )

    # 7. Fire Up Supervised Fine-Tuning (SFT) Trainer
    dataset = prepare_mock_dataset()
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )

    print("🚀 Initiating backpropagation training loops...")
    trainer.train()
    
    # 8. Save Optimized Trainable Parameter Adaptations
    print(f"💾 Saving trained adapter weights safely to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    print("🎉 Fine-tuning routine complete!")


if __name__ == "__main__":
    run_qlora_training()
