import base64
from collections import Counter
from whisper.tokenizer import get_tokenizer, Tokenizer

class RemappedTokenizer:
    #wraps the full gpt-2 tokenizer and remaps token IDs to a smaller set of english-only IDs
    
    def __init__(self, base: Tokenizer, id_map: dict[int, int]):
        self.base = base
        self.id_map = id_map
        self.rev_map = {v: k for k, v in id_map.items()}

        #remapping special tokens
        self.sot = id_map[base.sot]
        self.transcribe = id_map[base.transcribe]
        self.eot = id_map[base.eot]
        self.no_timestamps = id_map[base.no_timestamps]
        self.timestamp_begin = id_map[base.timestamp_begin]

        self.pad = id_map.get(base.special_tokens.get("pad"), None)

        self.sot_sequence = (self.sot, self.transcribe)

    def encode(self, text: str, **kwargs) -> list[int]:
        orig_ids = self.base.encoding.encode(text, **kwargs)
        return [self.id_map[t] for t in orig_ids if t in self.id_map]

    def decode(self, token_ids: list[int], **kwargs) -> str:

        orig_ids = [self.rev_map[t] for t in token_ids if t in self.rev_map]
        #removing timestamp tokens
        filtered = [t for t in orig_ids if t < self.base.timestamp_begin]
        return self.base.encoding.decode(filtered, **kwargs)

    @property
    def non_speech_tokens(self) -> tuple[int, ...]:
        original = self.base.non_speech_tokens
        return tuple(self.id_map[t] for t in original if t in self.id_map)


def build_remapped_tokenizer(train_split: list[tuple[str, str]]):

    #full gpt-2 tokenizer
    base_tok = get_tokenizer()

    counts = Counter()
    for _, text in train_split:
        counts.update(base_tok.encoding.encode(text))
    used = sorted(counts.keys())

    id_map = {old: new for new, old in enumerate(used)}
    next_id = len(used)

    specials = {
        base_tok.sot,
        base_tok.transcribe,
        base_tok.eot,
        base_tok.no_timestamps,
        base_tok.timestamp_begin,
    }
    for tok in specials:
        if tok not in id_map:
            id_map[tok] = next_id
            next_id += 1

    new_vocab_size = next_id

    remapped = RemappedTokenizer(base_tok, id_map)
    return remapped, new_vocab_size
