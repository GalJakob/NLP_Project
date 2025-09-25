### max bytes in a sentence:1120

from datasets import load_from_disk
dataset = load_from_disk("combined_asr_dataset")

# Check columns
print(dataset)

# Example: assume text is in column "sentence"
texts = dataset["sentence"]

# Compute max byte length
max_len = max(len(t.encode("utf-8")) for t in texts if t is not None)
print("Max byte length:", max_len)

# Show the longest example
longest = max(texts, key=lambda t: len(t.encode("utf-8")) if t is not None else -1)
print("Longest sentence:", longest)

