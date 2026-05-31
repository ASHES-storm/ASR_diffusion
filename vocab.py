import string


# Characters the model can predict
VOCAB = (
    ["<BLANK>", "<PAD>", "<UNK>"] +
    list(string.ascii_lowercase) +
    list(string.digits) +
    [" ", ".", ",", "?", "!", "'", "-"]
)

VOCAB_SIZE = len(VOCAB)


# Mappings
char2idx = {
    c: i
    for i, c in enumerate(VOCAB)
}

idx2char = {
    i: c
    for i, c in enumerate(VOCAB)
}


# Special tokens
BLANK_IDX = char2idx["<BLANK>"]
PAD_IDX = char2idx["<PAD>"]
UNK_IDX = char2idx["<UNK>"]


# Safety check for CTC
assert VOCAB[0] == "<BLANK>"


def text_to_ids(text):
    """
    Convert text -> token ids

    Example:
        "hello"
        ->
        [10, 7, 14, 14, 17]
    """

    text = text.lower()

    ids = []

    for char in text:

        if char in char2idx:
            ids.append(
                char2idx[char]
            )
        else:
            ids.append(
                UNK_IDX
            )

    return ids


def ids_to_text(ids):
    """
    Convert token ids -> text

    Example:
        [10, 7, 14, 14, 17]
        ->
        "hello"
    """

    chars = []

    for idx in ids:

        if idx in (
            BLANK_IDX,
            PAD_IDX,
            UNK_IDX
        ):
            continue

        chars.append(
            idx2char[idx]
        )

    return "".join(chars)
