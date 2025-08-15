import json
import sys
import jiwer
from tqdm import tqdm


def clean_id(utt_id):
    # Remove .wav and any numeric prefix before the first underscore
    utt_id = utt_id.replace(".wav", "")
    parts = utt_id.split("_")
    if parts[0].isdigit():
        return "_".join(parts[1:])
    return utt_id


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {clean_id(item["utt_id"]): item["text"].strip() for item in data}


def load_outputs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {clean_id(k): v["text"].strip() for k, v in data.items()}


def compute_metrics(references, hypotheses):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])

    refs = transformation(references)
    hyps = transformation(hypotheses)

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)

    return wer, cer


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 WER.py ground_truth.json output.json")
        sys.exit(1)

    gt_path = sys.argv[1]
    output_path = sys.argv[2]

    ground_truth = load_ground_truth(gt_path)
    output = load_outputs(output_path)

    # Match keys in both files
    common_ids = set(ground_truth.keys()).intersection(set(output.keys()))
    print(f"✅ Found {len(common_ids)} matching utterances.\n")

    if len(common_ids) == 0:
        print("❌ No matching utterance IDs found.")
        sys.exit(1)

    refs = [ground_truth[utt_id] for utt_id in common_ids]
    hyps = [output[utt_id] for utt_id in common_ids]

    wer, cer = compute_metrics(refs, hyps)
    print(f"\n🎯 Word Error Rate (WER): {wer * 100:.2f}%")
    print(f"✍️ Character Error Rate (CER): {cer * 100:.2f}%")


if __name__ == "__main__":
    main()
