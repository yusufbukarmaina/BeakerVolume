import os
import torch
from transformers import AutoProcessor, Trainer, TrainingArguments
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from config import Config
from data import load_dataset_split, detect_image_col, split_dataset, prepare

cfg = Config()

def collator(processor, image_col):
    def fn(examples):
        images = [ex[image_col] for ex in examples]
        texts = [cfg.QUESTION + " " + f"{ex['_volume']} mL" for ex in examples]

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return {k: v.to("cuda") for k, v in inputs.items()}
    return fn

def main():
    os.makedirs(cfg.QWEN_OUT, exist_ok=True)

    ds = load_dataset_split(cfg)
    image_col = detect_image_col(ds)
    train, val, _ = split_dataset(ds, cfg)

    train = prepare(train)
    val = prepare(val)

    processor = AutoProcessor.from_pretrained(cfg.QWEN_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.QWEN_MODEL)

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"],
                      task_type="CAUSAL_LM")

    model = get_peft_model(model, lora)
    model.to("cuda")

    args = TrainingArguments(
        output_dir=cfg.QWEN_OUT,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        fp16=cfg.FP16,
        logging_steps=20,
        save_steps=100,
        evaluation_strategy="steps",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator(processor, image_col)
    )

    trainer.train()
    trainer.save_model(cfg.QWEN_OUT)
    processor.save_pretrained(cfg.QWEN_OUT)

if __name__ == "__main__":
    main()
