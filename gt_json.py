import json

input_file = "test_transcripts.txt"
output_file = "ground_truth.json"

data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue  # skip malformed lines
        utt_id, text = parts
        data.append({"utt_id": utt_id, "text": text})

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(data)} entries to {output_file}")
