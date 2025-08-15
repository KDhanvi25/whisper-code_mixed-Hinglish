#!/bin/bash

mkdir -p data/test/segments_wav

while read utt_id wav_id start end; do
  duration=$(echo "$end - $start" | bc)
  wav_path=$(grep "^$wav_id " data/test/transcripts/wav_fixed.scp | awk '{print $2}')

  if [ -f "$wav_path" ]; then
    echo "Extracting $utt_id from $wav_id ($start-$end)..."
    ffmpeg -i "$wav_path" -ss "$start" -t "$duration" -ar 16000 -ac 1 "data/test/segments_wav/${utt_id}.wav" -y
  else
    echo "Warning: Audio file for $wav_id not found at $wav_path!"
  fi
done < data/test/transcripts/segments
