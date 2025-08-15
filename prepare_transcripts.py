import os

base_dir = "data/test"
transcript_dir = os.path.join(base_dir, "transcripts")
segments_file = os.path.join(transcript_dir, "segments")
txt_file = os.path.join(transcript_dir, "text")
output_file = "test_transcripts.txt"

# Load transcripts
txt_map = {}
with open(txt_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            utt_id, text = parts
            txt_map[utt_id] = text

# Build test_transcripts.txt
with open(segments_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        utt_id, wav_id, start, end = line.strip().split()
        audio_file = f"{utt_id}.wav"  # matches extracted files in segments_wav/
        text = txt_map.get(utt_id, "")
        if text:
            f_out.write(f"{audio_file}\t{text}\n")
        else:
            print(f"Warning: No transcript found for {utt_id}")

print(f"✅ Created {output_file} with references")
