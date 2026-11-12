"""Word timing: forced alignment of the transcript words to the audio.

An English CTC speech model (wav2vec2) gives, for every 20 ms frame, how likely
each letter is. Forced alignment then finds the most likely way to place the
known words of a line, in order, on those frames. It only places the words it is
given and never invents any. Digits are spelled out for the aligner only
("25" becomes "twenty five"); the word itself stays as written.
"""

import re
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import torch

SAMPLING_RATE: Final = 16_000
_ONES: Final = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS: Final = [
    "_",
    "_",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


class WordTime(NamedTuple):
    """Where one word was placed.

    Attributes:
        position: Position of the word in the line, from 0.
        start: Start in seconds, from the start of the audio passed in.
        end: End in seconds.
        score: Mean probability of the word's letters along the alignment, 0 to 1.
    """

    position: int
    start: float
    end: float
    score: float


def number_words(number: int) -> str:
    """Spell a whole number in English words, for example 125 -> one hundred twenty five.

    Args:
        number: A number from 0 to 999,999; longer numbers are read digit by digit.

    Returns:
        The number in words.
    """
    if number < 20:
        return _ONES[number]
    if number < 100:
        rest = number % 10
        return _TENS[number // 10] + (f" {_ONES[rest]}" if rest else "")
    if number < 1000:
        rest = number % 100
        return f"{_ONES[number // 100]} hundred" + (f" {number_words(rest)}" if rest else "")
    if number < 1_000_000:
        rest = number % 1000
        return f"{number_words(number // 1000)} thousand" + (
            f" {number_words(rest)}" if rest else ""
        )
    return " ".join(_ONES[int(digit)] for digit in str(number))


def spoken(word: str) -> str:
    """Return a normalised word as the aligner should hear it: digits spelled out.

    Args:
        word: A lower-case word of letters and digits, for example ``3rd``.

    Returns:
        Letters and spaces only, for example ``three rd``.
    """
    pieces = re.findall(r"\d+|[a-z]+", word)
    return " ".join(number_words(int(piece)) if piece.isdigit() else piece for piece in pieces)


def targets(words: list[str], vocab: dict[str, int]) -> tuple[list[int], list[int]]:
    """Return the aligner's target tokens and, for each, the word it belongs to.

    Words are separated by the model's word delimiter ``|``; a separator between
    two words belongs to no word (-1). Letters outside the vocabulary are skipped.

    Args:
        words: The line's words, as returned by ``transcripts.normalise``.
        vocab: The model's vocabulary: upper-case letters, ``'`` and ``|``.

    Returns:
        The token ids and the owner of each token.
    """
    ids: list[int] = []
    owners: list[int] = []
    delimiter = vocab["|"]
    for index, word in enumerate(words):
        if index:
            ids.append(delimiter)
            owners.append(-1)
        for part, piece in enumerate(spoken(word).split()):
            if part:
                ids.append(delimiter)
                owners.append(index)
            for letter in piece.upper():
                if letter in vocab:
                    ids.append(vocab[letter])
                    owners.append(index)
    return ids, owners


def align_words(
    log_probs: torch.Tensor,
    words: list[str],
    vocab: dict[str, int],
    blank: int,
    seconds_per_frame: float,
) -> list[WordTime] | None:
    """Place the words of one line on the frames of its audio.

    Args:
        log_probs: Log-probabilities from the CTC model, shape (frames, vocabulary).
        words: The line's words.
        vocab: The model's vocabulary.
        blank: Id of the CTC blank token.
        seconds_per_frame: Duration of one frame.

    Returns:
        One entry per word, in order; None if the words do not fit on the frames
        (the audio is too short for them) or a word has no letters to align.
    """
    import torchaudio.functional as F

    ids, owners = targets(words, vocab)
    if not ids or set(range(len(words))) - set(owners):
        return None
    try:
        aligned, scores = F.forced_align(
            log_probs.unsqueeze(0).float(), torch.tensor([ids], dtype=torch.int32), blank=blank
        )
    except RuntimeError:
        return None
    spans = F.merge_tokens(aligned[0], scores[0].exp())
    if len(spans) != len(ids):
        return None
    result = []
    for index in range(len(words)):
        own = [span for span, owner in zip(spans, owners, strict=True) if owner == index]
        result.append(
            WordTime(
                position=index,
                start=own[0].start * seconds_per_frame,
                end=own[-1].end * seconds_per_frame,
                score=float(np.mean([span.score for span in own])),
            )
        )
    return result


class WordAligner:
    """The pinned wav2vec2 CTC model, run in fp16, with forced alignment on top.

    Args:
        snapshot: Folder of the pinned ``facebook/wav2vec2-large-960h-lv60-self`` snapshot.
        device: Where the model runs.
    """

    def __init__(self, snapshot: Path, device: str = "cuda") -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = device
        self.processor: Any = Wav2Vec2Processor.from_pretrained(snapshot)
        model: Any = Wav2Vec2ForCTC.from_pretrained(snapshot)
        self.model = model.to(device).half().eval()
        self.vocab: dict[str, int] = self.processor.tokenizer.get_vocab()
        self.blank: int = self.processor.tokenizer.pad_token_id

    def align(self, audio: np.ndarray, words: list[str]) -> list[WordTime] | None:
        """Align the words of one line to its audio.

        Args:
            audio: Mono 16 kHz audio of the line.
            words: The line's words.

        Returns:
            One entry per word, with times relative to the start of ``audio``, or
            None if the alignment fails.
        """
        inputs = self.processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        values = inputs.input_values.to(self.device).half()
        with torch.inference_mode():
            logits = self.model(values).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        seconds_per_frame = len(audio) / SAMPLING_RATE / log_probs.shape[0]
        return align_words(log_probs, words, self.vocab, self.blank, seconds_per_frame)
