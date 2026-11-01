"""The buyer's own count of tokens. A seller's receipt is only its word, so what
gets paid is the smaller of the two counts, and a receipt far above this count
ends the session. The tokenizer is the model's own and public. Without it the
count falls back to the byte estimate, which the tolerance is wide enough for."""

from __future__ import annotations

import model_profiles as mp
from serve import estimate_tokens

MESSAGE_OVERHEAD = 8  # role markers; the rest of the template is in TOKEN_SLACK


class TokenCounter:
    def __init__(self, profile: mp.Profile | None = None):
        self.tokenizer = None
        if profile is not None:
            try:
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                self.tokenizer = Tokenizer.from_file(
                    hf_hub_download(profile.repo, "tokenizer.json", revision=profile.revision))
            except Exception:  # offline, or no such file: the estimate still bounds a seller
                pass

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return estimate_tokens(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def prompt(self, messages: list[dict]) -> int:
        return sum(self.count(str(m.get("content", ""))) + MESSAGE_OVERHEAD for m in messages)
