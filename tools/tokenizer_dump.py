#!/usr/bin/env python3
"""The oracle of `cancho-gpu tokenize` for tokenizer.json files: Hugging Face `tokenizers` (the library `transformers` uses),
printing the canonical form of `src/tokenize.cho` (docs/runtime.md section 9).

    tokenizer_dump.py <tokenizer.json> <corpus>

The corpus is records separated by a line holding only `%%`. It replaced `tools/tokenizer_dump.rs` (lex-rt's tokenizer) for
tokenizer.json files and agrees with every golden that dumper made (`docs/native.md` N2f). A GGUF tokenizer has no independent
reader here: see `tests/tokenizer/make_fixtures.py` and `scripts/regen-tokenizer-golden.sh`.
"""
import sys

from tokenizers import Tokenizer


def quoted(s):
    out = ['"']
    for b in s.encode("utf-8"):
        if b in (0x22, 0x5C):
            out.append("\\" + chr(b))
        elif b < 32 or b >= 127:
            out.append("\\x%02x" % b)
        else:
            out.append(chr(b))
    out.append('"')
    return "".join(out)


def main():
    try:
        tok = Tokenizer.from_file(sys.argv[1])
    except Exception as e:  # a tokenizer the library cannot read
        print("error: %s" % e, file=sys.stderr)
        sys.exit(1)
    corpus = open(sys.argv[2], encoding="utf-8", newline="").read()  # no newline translation: the corpus has \r\n
    print("tokenizer ids=%d" % tok.get_vocab_size(with_added_tokens=True))
    for record in corpus.split("\n%%\n"):
        ids = tok.encode(record, add_special_tokens=False).ids
        print("ids %d: %s" % (len(ids), " ".join(str(i) for i in ids)))
        print("dec %s" % quoted(tok.decode(ids, skip_special_tokens=False)))


main()
