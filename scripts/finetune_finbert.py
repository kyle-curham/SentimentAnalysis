#!/usr/bin/env python
"""Finetune FinBERT on Financial PhraseBank dataset."""
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import logging
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure output directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("models/finbert_finetuned", exist_ok=True)

def compute_metrics(pred):
    """Compute metrics for model evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Load dataset
    logger.info("Downloading dataset")
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        "text": dataset["train"]["sentence"],
        "label": dataset["train"]["label"]
    })
    
    # Save raw labels for model mapping
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    
    # Save to CSV
    df.to_csv("data/financial_phrasebank.csv", index=False)
    logger.info("Dataset saved to CSV")
    
    # Split the dataset into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    logger.info(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    
    # Load FinBERT model and tokenizer
    logger.info("Loading FinBERT model and tokenizer")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    # Prepare datasets for training
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].astype(int).tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].astype(int).tolist()
    
    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=128)
    val_encodings = tokenizer(val_texts, padding="max_length", truncation=True, max_length=128)
    
    # Create PyTorch datasets
    class FinancialDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
    
    train_dataset = FinancialDataset(train_encodings, train_labels)
    val_dataset = FinancialDataset(val_encodings, val_labels)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="models/finbert_finetuned",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train the model
    logger.info("Starting model training")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating model")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save the model
    logger.info("Saving finetuned model")
    model.save_pretrained("models/finbert_finetuned/final")
    tokenizer.save_pretrained("models/finbert_finetuned/final")
    
    logger.info("Finetuning completed successfully")

if __name__ == "__main__":
    main() 