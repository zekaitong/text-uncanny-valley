# Dictionary Repair

`segment_repair_spacing_clean.py` repairs spacing-fragmented JSONL `lines`
using only an external English dictionary. It contains no benchmark-specific
fallback vocabulary. The included `en_US.dic` is the dictionary snapshot used
for the reported repair experiment.

```bash
python rebuttal/dictionary_repair/segment_repair_spacing_clean.py \
  --inputs data/legal_docs_w5.jsonl \
  --dictionary rebuttal/dictionary_repair/en_US.dic \
  --output-dir outputs/repaired
```

`configs/legal_gemini_example.yaml` shows the downstream evaluation entry.
No repaired documents, model responses, or aggregate result files are included.
Generated outputs should be written to an ignored output directory.

