import re
from collections import Counter

input_file = "test_transcripts.txt"
output_file = "prompt_words.txt"

# Read all text after the filename
with open(input_file, "r", encoding="utf-8") as f:
    all_text = " ".join([line.strip().split("\t", 1)[1] for line in f if "\t" in line])

# Tokenize into words (Devanagari + Latin letters)
words = re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+', all_text)

# Filter: length >= 2 to drop single phones like क, ह
words = [w for w in words if len(w) >= 2]

# Count frequencies
counter = Counter(words)
top_words = [w for w, _ in counter.most_common(100)]

# Save top 100 for manual review
with open(output_file, "w", encoding="utf-8") as f:
    f.write(" ".join(top_words))

print("✅ Saved cleaned top 100 words to prompt_words.txt")
