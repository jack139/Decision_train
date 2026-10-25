from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments

LLM_PATH = "../lm_model/unsloth_Qwen3.5-0.8B"
DATASET = "./data/typed-decisions"

model, tokenizer = FastDecisionModel.from_pretrained(
    model_name = LLM_PATH,
    max_seq_length = 2048,
    load_in_4bit = True,
    local_files_only = True,
)

model = FastDecisionModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

dataset = load_dataset(DATASET, "all", split = "train")
items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions")

train_items, eval_items = FastDecisionModel.split_holdout(items, seed = 3407)
print(f"{len(train_items)} training rows, {len(eval_items)} held out")


trainer = DecisionTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_items,
    eval_dataset = eval_items,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 10,
        # num_train_epochs = 2, # Set this for 1 full training run.
        max_steps = 240,
        learning_rate = 2e-5,
        lr_scheduler_type = "cosine",
        weight_decay = 0.01,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        eval_strategy = "epoch",
        logging_steps = 1,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
        seed = 3407,
    ),
)
trainer_stats = trainer.train()