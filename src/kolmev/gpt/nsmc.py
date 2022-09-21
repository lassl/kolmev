from argparse import ArgumentParser
from functools import partial
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import _torch_collate_batch


def get_args():
    parser = ArgumentParser()
    data_args_group = parser.add_argument_group("data")
    data_args_group.add_argument("--pretrained_model_name_or_path", required=True, type=str)
    data_args_group.add_argument("--max_sequence_length", default=128, type=int)

    train_args_group = parser.add_argument_group("train")
    train_args_group.add_argument(
        "--evaluation_strategy",
        choices=["no", "steps", "epoch"],
        type=str,
        default="epoch",
    )
    train_args_group.add_argument("--save_strategy", choices=["no", "epoch", "steps"], type=str, default="no")
    train_args_group.add_argument("--per_device_train_batch_size", type=int, default=32)
    train_args_group.add_argument("--per_device_eval_batch_size", type=int, default=32)
    train_args_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_args_group.add_argument("--eval_accumulation_steps", type=int, default=1)
    train_args_group.add_argument("--learning_rate", type=float, default=5e-5)
    train_args_group.add_argument("--weight_decay", type=float, default=0.01)
    train_args_group.add_argument("--max_grad_norm", type=float, default=1.0)
    train_args_group.add_argument("--num_train_epochs", type=int, default=5)
    train_args_group.add_argument("--seed", type=int, default=42)
    train_args_group.add_argument("--bf16", action="store_true")
    train_args_group.add_argument("--fp16", action="store_true")
    train_args_group.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    return args


def get_features(records, tokenizer, max_length):
    features = tokenizer(
        text=records["document"],
        padding=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_length=True,
    )
    return features


def batchify(list_of_samples: List[Dict[str, Union[int, List[int]]]], tokenizer):
    list_of_input_ids = [sample["input_ids"] for sample in list_of_samples]
    list_of_labels = [sample["label"] for sample in list_of_samples]
    list_of_lengths = [sample["length"] for sample in list_of_samples]
    max_length = max(list_of_lengths)

    input_ids: torch.Tensor = _torch_collate_batch(list_of_input_ids, tokenizer)
    attention_mask: torch.Tensor = torch.ones((len(list_of_input_ids), max_length))
    for idx, length in enumerate(list_of_lengths):
        attention_mask[idx, length:] = 0.0
    labels = torch.tensor(list_of_labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_accuracy(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    result = metric.compute(predictions=predictions, references=labels)
    return result


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path)
    if not model.config.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset(
        "nsmc",
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    args.max_sequence_length = min(args.max_sequence_length, model.config.n_ctx)
    train_dataset = train_dataset.map(
        lambda records: get_features(records, tokenizer, args.max_sequence_length),
        batched=True,
        remove_columns=["id", "document"],
    )
    test_dataset = test_dataset.map(
        lambda records: get_features(records, tokenizer, args.max_sequence_length),
        batched=True,
        remove_columns=["id", "document"],
    )
    batchify_ = partial(batchify, tokenizer=tokenizer)

    with TemporaryDirectory() as tmp:
        training_args = TrainingArguments(
            output_dir=tmp,
            evaluation_strategy=args.evaluation_strategy,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_accumulation_steps=args.eval_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            num_train_epochs=args.num_train_epochs,
            save_strategy=args.save_strategy,
            seed=args.seed,
            bf16=args.bf16,
            fp16=args.fp16,
            group_by_length=True,
            length_column_name="length",
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=batchify_,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_accuracy,
        )
        trainer.train()


if __name__ == "__main__":
    main()
