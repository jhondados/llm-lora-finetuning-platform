"""QLoRA fine-tuning with PEFT."""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

def create_qlora_model(model_id: str = "google/gemma-2-9b-it"):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    return get_peft_model(model, lora_config)

def train(model_id: str, dataset, output_dir: str):
    model = create_qlora_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=2e-4, bf16=True, logging_steps=10,
        save_strategy="epoch", optim="paged_adamw_32bit", lr_scheduler_type="cosine")
    trainer = SFTTrainer(model=model, train_dataset=dataset, args=args, tokenizer=tokenizer,
        dataset_text_field="text", max_seq_length=4096, packing=True)
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
