####
# asr_output percentiles (bytes): {50: 79.0, 90: 167.0, 95: 218.0, 99: 402.0, 99.5: 482.0, 100: 740.0} | max: 740
# sentence   percentiles (bytes): {50: 86.0, 90: 198.0, 95: 274.0, 99: 542.0, 99.5: 632.0099999999948, 100: 1120.0} | max: 1120

# Suggested max_length (inputs): 404 (rounded: 408)
# Suggested generation_max_length: 544 (rounded: 544)

from datasets import load_from_disk, DatasetDict
import numpy as np
ds = load_from_disk("combined_asr_dataset")

def byte_len(s):
    return len(s.encode("utf-8")) if s is not None else 0

def percentiles_for(ds, column, qs=(50, 90, 95, 99, 99.5, 100)):
    texts = ds[column]                         # simple, like your template
    lens  = np.array([byte_len(t) for t in texts], dtype=np.int32)
    pct   = {q: float(np.percentile(lens, q)) for q in qs}
    return lens, pct

# Compute percentiles for your two columns
in_lens,  in_pct  = percentiles_for(ds, "asr_output")
tgt_lens, tgt_pct = percentiles_for(ds, "sentence")

print("asr_output percentiles (bytes):", in_pct,  "| max:", int(in_lens.max()))
print("sentence   percentiles (bytes):", tgt_pct, "| max:", int(tgt_lens.max()))

# Suggested caps (~99th percentile + 2 for specials)
cap_in  = int(in_pct[99]  + 2)
cap_tgt = int(tgt_pct[99] + 2)

# (Optional) round to multiple of 8 if you'll use pad_to_multiple_of=8
round8 = lambda x: int(((x + 7) // 8) * 8)
print("\nSuggested max_length (inputs):", cap_in,  f"(rounded: {round8(cap_in)})")
print("Suggested generation_max_length:", cap_tgt, f"(rounded: {round8(cap_tgt)})")

# Show the single longest examples (like your template)
i_in  = int(in_lens.argmax())
i_tgt = int(tgt_lens.argmax())
print("\nLongest asr_output bytes:", in_lens[i_in]);  print(ds[i_in]["asr_output"])
print("\nLongest sentence bytes  :", tgt_lens[i_tgt]); print(ds[i_tgt]["sentence"])
