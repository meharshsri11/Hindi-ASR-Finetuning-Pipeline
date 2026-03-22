import json
import torch
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # THE FIX: Forcefully slice massive transcripts so they never exceed the 448 token limit
    batch["labels"] = processor(text=batch["transcription"]).input_ids[:448]
    return batch

def main():
    print("1. Loading Processed Dataset...")
    DATA_PATH = "data/processed/dataset.json"
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Just 20 samples to keep it ultra-fast
    small_data = data[:20]
    
    hf_dataset = Dataset.from_list(small_data)
    hf_dataset = hf_dataset.rename_column("audio_url", "audio")
    hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=16000))

    print("2. Loading Whisper Model & Processor...")
    model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_id, language="Hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    print("3. Extracting Features...")
    prepared_dataset = hf_dataset.map(
        lambda x: prepare_dataset(x, processor), 
        remove_columns=hf_dataset.column_names,
        num_proc=1
    )

    print("4. Setting up Training Configuration...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Simplified training args for a guaranteed run
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-finetuned-hindi",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=10,
        max_steps=5,  # 5 steps only
        fp16=torch.cuda.is_available(), 
        save_steps=5,
        logging_steps=1,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=prepared_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("\n5. Starting Fine-Tuning Process (Test Mode - 5 Steps Only)...")
    trainer.train()
    
    print("\n✅ Training Script executed successfully! Model saved in './whisper-finetuned-hindi'")

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

if __name__ == "__main__":
    main()