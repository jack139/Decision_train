
import json
import torch
from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments


LLM_PATH = "../lm_model/unsloth_Qwen3.5-0.8B"
#LLM_PATH = "../lm_model/unsloth_Qwen3.5-4B"
DATASET = "./data/typed-decisions"

# A decision model doesn't write text. It reads an input, looks at the options you give it, and picks one with a probability. FastDecisionModel turns an LLM into one: it adds a small head that reads every question about an input in one pass and scores each option.

model, tokenizer = FastDecisionModel.from_pretrained(
    model_name = LLM_PATH,
    max_seq_length = 2048, # Longest input. Long inputs keep the question and options.
    load_in_4bit = True, # 4 bit quantization to reduce memory
)

model = FastDecisionModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
)


# Data Prep

dataset = load_dataset(DATASET, "all", split = "train")
print(dataset[0]["state"][:500])
print(json.dumps(json.loads(dataset[0]["questions"]), indent = 2)[:1000])
print(dataset[0]["gold"])


items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions")

train_items, eval_items = FastDecisionModel.split_holdout(items, seed = 3407)
print(f"{len(train_items)} training rows, {len(eval_items)} held out")

# Let's check accuracy before training. The head is new, so it's about the same as guessing (about 30% accuracy in our runs).
FastDecisionModel.evaluate(model, tokenizer, eval_items)



# Train the model

trainer = DecisionTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_items,
    eval_dataset = eval_items,
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 10,
        # num_train_epochs = 2, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        lr_scheduler_type = "cosine",
        weight_decay = 0.01,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        eval_strategy = "epoch",
        logging_steps = 10,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
        seed = 3407,
    ),
)


# Show current memory stats

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set trainer.train(resume_from_checkpoint = True)
trainer_stats = trainer.train()


# Show final memory and time stats
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



# Inference

FastDecisionModel.calibrate(model, tokenizer, eval_items)

# Let's check accuracy on the dataset's test split, which the model never saw. We got 75% to 78% after 60 steps:

test = load_dataset(DATASET, "all", split = "test")
test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
FastDecisionModel.evaluate(model, tokenizer, test_items)


# Now let's make some decisions!
FastDecisionModel.for_inference(model)
answers = FastDecisionModel.predict(
    model,
    tokenizer,
    "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, errors",
                "sales": "pricing, new plans",
            },
        },
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "today"],
        },
    },
)
for name, result in answers.items():
    print(name, result["answer"], {k: round(v, 3) for k, v in result["probabilities"].items()})


# Saving, loading finetuned models
model.save_pretrained("qwen_lora")  # Local saving
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving



# Now if you want to load the model we just saved, set False to True:
if False:
    from unsloth import FastDecisionModel
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name = "qwen_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )


# Saving to float16
# Merge to 16bit
if False:
    model.save_pretrained_merged("qwen_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("HF_USERNAME/qwen_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")


