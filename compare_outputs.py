import json
import random

# --- Load JSON files ---
with open("whisper_outputs_baseline.json", "r", encoding="utf-8") as f:
    baseline_outputs = json.load(f)

with open("whisper_outputs_biased.json", "r", encoding="utf-8") as f:
    biased_outputs = json.load(f)

# --- Common keys ---
common_keys = list(set(baseline_outputs.keys()) & set(biased_outputs.keys()))
print(f"\n✅ Found {len(common_keys)} common utterances.")

# --- Sample 10 for comparison ---
sample_keys = random.sample(common_keys, 10)

print("\n🔍 Comparing 10 Random Utterances:\n")
for utt_id in sample_keys:
    print(f"🆔 {utt_id}")
    print(f"  Baseline : {baseline_outputs[utt_id]}")
    print(f"  Biased   : {biased_outputs[utt_id]}")
    print("-" * 80)
