import string
# characters your model can predict
VOCAB = (
    ["<BLANK>", "<PAD>", "<UNK>"] +   # special tokens
    list(string.ascii_lowercase) +    # a-z
    list(string.digits) +             # 0-9
    [" ", ".", ",", "?", "!", "'", "-"]
)
VOCAB_SIZE = len(VOCAB)

# mappings
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for i, c in enumerate(VOCAB)}
#special tokens mapping
BLANK_IDX = char2idx["<BLANK>"]
PAD_IDX = char2idx["<PAD>"]
UNK_IDX = char2idx["<UNK>"]
#safety check
assert VOCAB[0] == "<BLANK>"
