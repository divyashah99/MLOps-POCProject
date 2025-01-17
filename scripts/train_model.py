
import os
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def download_dataset_from_gcs(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded dataset to {destination_file_name}")

def train_and_save_model():
    BUCKET_NAME = "mlops-poc-project-bucket"
    DATASET_PATH = "data/dataset.csv"
    LOCAL_PATH = "data/training_dataset.csv"

    # Step 1: Download Dataset
    download_dataset_from_gcs(BUCKET_NAME, DATASET_PATH, LOCAL_PATH)

    # Step 2: Load and Preprocess Dataset
    dataset = load_dataset("csv", data_files=LOCAL_PATH)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Step 3: Define Model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Step 4: Training
    training_args = TrainingArguments(
        output_dir="/model_output",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trainer.train()

    # Step 5: Save Model to GCS
    model_path = "model/trained_model"
    trainer.save_model(model_path)
    print("Model trained and saved locally at", model_path)

if __name__ == "__main__":
    train_and_save_model()
