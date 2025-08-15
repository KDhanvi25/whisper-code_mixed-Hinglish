import os
import json
import torch
import soundfile as sf
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# === Paths ===
AUDIO_DIR = "data/test/segments_wav"
OUTPUT_FILE = "finetune_outputs.json"
MODEL_DIR = "ckpts/whisper"  # Top-level folder with your best fine-tuned model

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Load model & processor ===
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
processor = WhisperProcessor.from_pretrained(MODEL_DIR)

results = {}

for fname in os.listdir(AUDIO_DIR):
    if fname.endswith(".wav"):
        audio_path = os.path.join(AUDIO_DIR, fname)

        # Load audio
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        if sr != 16000:
            audio = resample_poly(audio, 16000, sr)

        # Preprocess
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(inputs, max_new_tokens=256)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Save results
        results[fname] = {
            "text": transcription
        }

# Write to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Saved transcripts to {OUTPUT_FILE}")
