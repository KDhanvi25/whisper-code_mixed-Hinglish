#!/usr/bin/env python3
import argparse, os, json, math, shutil, sys
from collections import defaultdict, OrderedDict
from pathlib import Path

def read_kaldi_kv(path):
    d = OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            k, v = line.rstrip('\n').split(maxsplit=1)
            d[k] = v
    return d

def read_text(path):
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.rstrip('\n').split()
            d[parts[0]] = " ".join(parts[1:])
    return d

def read_segments(path):
    segs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            sid, rid, s, e = line.rstrip('\n').split()
            segs.append((sid, rid, float(s), float(e)))
    return segs

def seconds_to_hours(s): return s / 3600.0

def safe_abspath(train_dir, wav_path):
    # If relative or just a filename, join with train_dir
    p = Path(wav_path)
    if not p.is_absolute():
        p = Path(train_dir) / p
    return str(p.resolve())

def get_wav_duration_seconds(path):
    # Try soundfile; fall back to wave
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    import wave
    with wave.open(path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def write_split_kaldi(split_dir, examples):
    os.makedirs(split_dir, exist_ok=True)
    # Build per-split wav.scp (unique by recording id), text, segments/utt2spk
    wav_map = OrderedDict()
    text_map = OrderedDict()
    seg_lines = []
    utt2spk = OrderedDict()

    for ex in examples:
        wav_map.setdefault(ex['rec_id'], ex['wav_path'])
        text_map[ex['utt_id']] = ex['text']
        utt2spk[ex['utt_id']] = ex['spk']
        if ex['start'] is not None:
            seg_lines.append(f"{ex['utt_id']} {ex['rec_id']} {ex['start']:.3f} {ex['end']:.3f}")

    # Write wav.scp
    with open(Path(split_dir)/'wav.scp', 'w', encoding='utf-8') as w:
        for rid, p in wav_map.items():
            w.write(f"{rid} {p}\n")
    # Write text
    with open(Path(split_dir)/'text', 'w', encoding='utf-8') as w:
        for uid, t in text_map.items():
            w.write(f"{uid} {t}\n")
    # Write segments if we used segments or created virtual ones
    if seg_lines:
        with open(Path(split_dir)/'segments', 'w', encoding='utf-8') as w:
            for line in seg_lines:
                w.write(line + "\n")
    # utt2spk & spk2utt
    with open(Path(split_dir)/'utt2spk', 'w', encoding='utf-8') as w:
        for uid, spk in utt2spk.items():
            w.write(f"{uid} {spk}\n")
    spk2utt = defaultdict(list)
    for uid, spk in utt2spk.items():
        spk2utt[spk].append(uid)
    with open(Path(split_dir)/'spk2utt', 'w', encoding='utf-8') as w:
        for spk, utts in spk2utt.items():
            w.write(f"{spk} {' '.join(utts)}\n")

    # Also drop a JSONL manifest
    with open(Path(split_dir)/'manifest.jsonl', 'w', encoding='utf-8') as w:
        for ex in examples:
            rec = dict(
                id=ex['utt_id'],
                audio=ex['wav_path'],
                text=ex['text'],
                start=ex['start'],
                duration=ex['duration']
            )
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True, help="Folder containing wav.scp, text, transcripts/, etc.")
    ap.add_argument('--transcripts', required=False, help="Transcripts folder if your code needs it; not required here.")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--train_hours', type=float, required=True)
    ap.add_argument('--dev_hours', type=float, required=True)
    args = ap.parse_args()

    train_dir = args.train_dir
    out_dir = args.out_dir

    wav_scp = Path(train_dir)/'transcripts/wav.scp'
    text_path = Path(train_dir)/'transcripts/text'
    seg_path = Path(train_dir)/'transcripts/segments'

    if not wav_scp.exists():
        sys.exit(f"ERROR: {wav_scp} not found")
    if not text_path.exists():
        sys.exit(f"ERROR: {text_path} not found")

    wav_map = read_kaldi_kv(wav_scp)
    # Normalize paths
    for rid in list(wav_map.keys()):
        wav_map[rid] = safe_abspath(train_dir, wav_map[rid])

    text_map = read_text(text_path)
    have_segments = seg_path.exists()
    segments = read_segments(seg_path) if have_segments else None

    examples = []
    missing = 0

    if have_segments:
        # Build from segments
        for sid, rid, s, e in segments:
            if rid not in wav_map:
                missing += 1
                continue
            if sid not in text_map:
                missing += 1
                continue
            wav_path = wav_map[rid]
            duration = max(0.0, e - s)
            if duration <= 0:
                continue
            examples.append(dict(
                utt_id=sid,
                rec_id=rid,
                wav_path=wav_path,
                start=s, end=e,
                duration=duration,
                text=text_map[sid],
                spk=rid  # simple default: speaker == recording
            ))
    else:
        # Whole-file examples
        for rid, wav_path in wav_map.items():
            # If no text for recording id, skip
            if rid not in text_map:
                continue
            try:
                dur = get_wav_duration_seconds(wav_path)
            except Exception as ex:
                print(f"WARNING: failed duration for {wav_path}: {ex}")
                continue
            examples.append(dict(
                utt_id=rid,
                rec_id=rid,
                wav_path=wav_path,
                start=None, end=None,
                duration=dur,
                text=text_map[rid],
                spk=rid
            ))

    if not examples:
        sys.exit("No usable utterances found. Check wav.scp paths and text/segments IDs.")

    # Keep input order (already preserved) but you could sort by duration if you want longest first
    total_hours = seconds_to_hours(sum(e['duration'] for e in examples))
    print(f"Found {len(examples)} utterances, total {total_hours:.2f} hours.")

    # Greedy fill train then dev
    target_train = args.train_hours
    target_dev = args.dev_hours
    train, dev = [], []
    h_train = 0.0
    h_dev = 0.0

    for ex in examples:
        h = seconds_to_hours(ex['duration'])
        if h_train + h <= target_train + 1e-6:
            train.append(ex); h_train += h
        elif h_dev + h <= target_dev + 1e-6:
            dev.append(ex); h_dev += h
        # else ignore extras

    if not train:
        sys.exit("After selection, train split is empty — lower --train_hours or check input.")
    if target_dev > 0 and not dev:
        print("NOTE: dev split ended up empty; consider lowering --train_hours or raising --dev_hours.")

    # Write out
    split_train = Path(out_dir)/'train'
    split_dev   = Path(out_dir)/'dev'
    write_split_kaldi(split_train, train)
    if dev:
        write_split_kaldi(split_dev, dev)

    print(f"✅ Wrote {len(train)} train utts (~{h_train:.2f} h) to {split_train}")
    if dev:
        print(f"✅ Wrote {len(dev)} dev utts (~{h_dev:.2f} h) to {split_dev}")
    if missing:
        print(f"⚠️ Skipped {missing} items due to missing IDs or paths.")

if __name__ == "__main__":
    main()
