"""ECC-001 data preparation: WikiText-2 raw (public) -> uint16 token files for the harness, with the published byte-level BPE (vocabulary 2,048).

    python prepare_data.py --train train-00000-of-00001.parquet --valid validation-00000-of-00001.parquet --tokenizer tokenizer.json --out DIR

Source: Hugging Face dataset Salesforce/wikitext, revision b08601e04326c79dfdd32d625aee71d232d685c3, config wikitext-2-raw-v1, files
train-00000-of-00001.parquet (sha256 e83889baabc49707...) and validation-00000-of-00001.parquet (sha256 204929b7ff9d6184...).
Text units: every row of the 'text' column, stripped; empty rows dropped; validation rows identical to a train row dropped (exact overlap removal);
joined with '\\n' plus a final '\\n', UTF-8.  Expected results (verified 2026-10-17):
  raw text   train sha256 1ac2aed33ca7f2ec..., validation sha256 e9d895b89a83002d...
  tokens     en_train.bin 3,860,300 tokens sha256 dba66a84c4c39397..., en_validation.bin 401,875 tokens sha256 5d50fda52f1d05ed...
The tokenizer (tokenizer.json, sha256 ee98f08a2af1f3ef...) was trained on WikiText-2 train, an Indonesian Wikipedia sample and TinyStories train;
never on validation text.  Requires pyarrow and tokenizers.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

EXPECT = {'en_train.bin': 'dba66a84c4c3939786ca3f772104ce56b421f54071b5cb73fa9535ce3e2790fb',
          'en_validation.bin': '5d50fda52f1d05ed22c8e1a5453b65e36a2a78e0f62a36d448375ed0f447c39c'}


def units(path):
    return [s.strip() for s in pq.read_table(path)['text'].to_pylist() if s.strip()]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--train'); p.add_argument('--valid'); p.add_argument('--tokenizer'); p.add_argument('--out')
    a = p.parse_args(); tok = Tokenizer.from_file(a.tokenizer); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = units(a.train); seen = set(tr); va = [s for s in units(a.valid) if s not in seen]
    for rows, name in ((tr, 'en_train.bin'), (va, 'en_validation.bin')):
        text = '\n'.join(rows) + '\n'
        ids = tok.encode(text, add_special_tokens=False).ids
        arr = np.asarray(ids, dtype=np.uint16); assert arr.max() < 2048 and tok.decode(ids) == text, 'tokenizer round trip failed'
        (out / name).write_bytes(arr.tobytes()); h = hashlib.sha256(arr.tobytes()).hexdigest()
        print(name, len(arr), 'tokens; raw sha256', hashlib.sha256(text.encode('utf-8')).hexdigest(), '; token sha256', h, '; matches', h == EXPECT[name])


if __name__ == '__main__':
    main()
