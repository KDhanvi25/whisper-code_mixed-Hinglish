import whisper
import os, json
import torch

AUDIO_DIR = "data/test/segments_wav"
OUTPUT_FILE = "whisper_outputs.json"

# Force GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("small").to(device)  # "small" can be changed to "medium"

results = {}
for fname in os.listdir(AUDIO_DIR):
    if fname.endswith(".wav"):
        audio_path = os.path.join(AUDIO_DIR, fname)
        print(f"Transcribing {fname}...")
        result = model.transcribe(audio_path, language=None, fp16=(device=="cuda"))
        results[fname] = {
            "text": result["text"],
            "segments": result["segments"]
        }

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Saved transcripts to {OUTPUT_FILE}")
