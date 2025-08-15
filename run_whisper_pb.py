import whisper
import os, json
import torch

AUDIO_DIR = "data/test/segments_wav"
OUTPUT_FILE = "whisper_outputs_biased.json"

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model
model = whisper.load_model("small").to(device)

# Load hybrid prompt words
with open("prompt_words.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read().strip()

results = {}
for fname in sorted(os.listdir(AUDIO_DIR)):
    if fname.endswith(".wav"):
        audio_path = os.path.join(AUDIO_DIR, fname)
        print(f"Transcribing {fname} with prompt biasing...")
        result = model.transcribe(
            audio_path,
            language=None,
            fp16=(device == "cuda"),
            initial_prompt=prompt_text
        )
        results[fname] = {
            "text": result["text"],
            "segments": result["segments"]
        }

# Save predictions
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Saved biased transcripts to {OUTPUT_FILE}")
