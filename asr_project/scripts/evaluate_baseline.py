import torch
import re
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
from tqdm import tqdm

def clean_text(text):
    """Removes punctuation and normalizes spaces to fix artificial WER inflation."""
    # Remove standard punctuations including Hindi purna viram (|)
    text = re.sub(r'[.,!?।\-\'\"]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate_baseline():
    print("1. Loading FLEURS Hindi Test Dataset...")
    # Use datasets==2.19.0 to allow trust_remote_code
    fleurs_test = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    fleurs_test = fleurs_test.select(range(50)) 
    print(f"Loaded {len(fleurs_test)} samples for evaluation.")

    print("\n2. Loading Whisper-Small Model & Processor...")
    model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_id)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    
    wer_metric = evaluate.load("wer")

    print("\n3. Running Inference and Calculating Normalized WER...")
    predictions = []
    references = []

    for item in tqdm(fleurs_test, desc="Evaluating"):
        audio = item["audio"]
        
        # APPLYING THE FIX: Clean the reference text
        reference_text = clean_text(item["transcription"])
        
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, 
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="hi", task="transcribe")
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # APPLYING THE FIX: Clean the prediction text
        transcription = clean_text(transcription)
        
        predictions.append(transcription)
        references.append(reference_text)

    wer = wer_metric.compute(predictions=predictions, references=references)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS (WITH PUNCTUATION FIX)")
    print("="*50)
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")
    print("\nSample Output Comparison:")
    print(f"Reference : {references[0]}")
    print(f"Prediction: {predictions[0]}")

if __name__ == "__main__":
    evaluate_baseline()