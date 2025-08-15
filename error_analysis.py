import sys
import json
import unicodedata
import collections
from jiwer import cer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, process_words
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(text):
    transform = Compose([
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip()
    ])
    return transform(text)

def script_category(c):
    # Classify character into script
    name = unicodedata.name(c, "")
    if "DEVANAGARI" in name:
        return "Devanagari"
    elif "LATIN" in name:
        return "Latin"
    else:
        return "Other"

def analyze_char_confusions(refs, hyps):
    substitutions = collections.Counter()
    script_errors = collections.Counter()
    all_ref_chars, all_hyp_chars = [], []

    for ref, hyp in zip(refs, hyps):
        ref_chars = list(ref)
        hyp_chars = list(hyp)

        # Align by Levenshtein distance (simple equal length truncation for now)
        min_len = min(len(ref_chars), len(hyp_chars))
        for r, h in zip(ref_chars[:min_len], hyp_chars[:min_len]):
            if r != h:
                substitutions[(r, h)] += 1
                if script_category(r) != script_category(h):
                    script_errors[(r, h)] += 1
            all_ref_chars.append(r)
            all_hyp_chars.append(h)

    return substitutions, script_errors, all_ref_chars, all_hyp_chars

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 error_analysis.py ground_truth.json baseline_outputs.json")
        return

    gt_data = load_json(sys.argv[1])
    pred_data = load_json(sys.argv[2])

    gt_dict = {x["utt_id"]: x["text"] for x in gt_data}
    pred_dict = {k: v["text"] for k, v in pred_data.items()}

    # Fix any ID mismatch like leading numbers in GT
    common_keys = []
    for utt_id in pred_dict:
        trimmed = utt_id.split("_", 1)[-1]
        match = next((gt_id for gt_id in gt_dict if trimmed in gt_id), None)
        if match:
            common_keys.append((match, utt_id))

    refs = [normalize(gt_dict[gt]) for gt, _ in common_keys]
    hyps = [normalize(pred_dict[pred]) for _, pred in common_keys]

    print(f"\n✅ Found {len(common_keys)} matching utterances.\n")

    # Compute WER breakdown
    overall = process_words(refs, hyps)
    print("📊 Word-level Error Breakdown:")
    print(f"  Substitutions: {overall.substitutions}")
    print(f"  Insertions:    {overall.insertions}")
    print(f"  Deletions:     {overall.deletions}")
    print(f"  Hits:          {overall.hits}")
    print(f"\n🎯 WER: {overall.wer * 100:.2f}%")

    # Compute CER
    cer_score = cer(refs, hyps)
    print(f"✍️ CER: {cer_score * 100:.2f}%")

    # Character-level confusion analysis
    char_conf, script_conf, all_ref_chars, all_hyp_chars = analyze_char_confusions(refs, hyps)

    print("\n🔡 Top 10 Character Substitutions:")
    for (r, h), count in char_conf.most_common(10):
        print(f"  {r} → {h}: {count} times")

    print("\n🔣 Top 5 Script Confusion Pairs:")
    for (r, h), count in script_conf.most_common(5):
        print(f"  {r} ({script_category(r)}) → {h} ({script_category(h)}): {count} times")


if __name__ == "__main__":
    main()
