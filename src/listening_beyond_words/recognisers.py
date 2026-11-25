"""The recognisers of the word check: Whisper large-v3 with faster-whisper, and wav2vec2.

Pilot protocol, section "Clip versions", bullet "Word check"; the settings are in
``configs/pilot/data.yaml`` (``word_check.whisper`` and ``word_check.wav2vec2``) and the
pinned snapshots in ``configs/pilot/models.yaml``. Both recognisers read the same array,
a clip's stored 16-bit samples divided by 32,768 as float32 (``check_audio``), and both
load offline from the pinned snapshots; jobs never download.

- Whisper (``Whisper``): faster-whisper 1.2.1 runs Whisper large-v3 in float16 with
  exactly the settings of the ``whisper`` block (``WhisperSettings``): English, beam 5,
  one temperature of 0, no earlier text, no voice-activity filter, and none of the
  library's rules that drop or re-decode the text of a window. A window is the 30 s of
  audio that Whisper reads at once. The other arguments of ``transcribe()`` cannot
  change the text with these settings (``WHISPER_INERT``). Each reading records its
  windows and counts those that the library's default silence rule would have dropped.
- wav2vec2 (``Wav2vec2``): the pinned CTC model in float32, its own precision, one clip
  at a time, in evaluation mode, decoded greedily: the most likely symbol in each 20 ms
  frame (``greedy_ctc``). ``alignment.WordAligner``, which placed the pinned word times
  in float16, stays as it is.

torch, transformers, faster-whisper and CTranslate2 are imported inside the functions
that use them, torch before faster-whisper: CTranslate2 loads cuBLAS by name when it
first needs it and finds the copy that PyTorch loaded. A reading holds participant text;
it stays in the run folder on OMNI, and the repr of ``Reading`` leaves it out.
"""

import copy
import itertools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Final, Protocol

import numpy as np

from listening_beyond_words.runinfo import StepError, deterministic_torch
from listening_beyond_words.views import SAMPLING_RATE

# Whisper -----------------------------------------------------------------------------

WHISPER_FILES: Final = (
    "model.bin",
    "config.json",
    "tokenizer.json",  # without it, faster-whisper would fetch a tokenizer from the Hub
    "preprocessor_config.json",  # without it, faster-whisper would build 80 mel bands
    "vocabulary.json",
)
MEL_BANDS: Final = 128  # the features of Whisper large-v3
# faster-whisper 1.2.1's defaults (WhisperModel.transcribe): a window whose no-speech
# probability is above the first loses all its text, unless its mean log-probability is
# above the second (transcribe.py, generate_segments).
DEFAULT_NO_SPEECH_THRESHOLD: Final = 0.6
DEFAULT_LOG_PROB_THRESHOLD: Final = -1.0
# The keys of the whisper block, in the order of data.yaml: compute_type goes to
# WhisperModel, every other key to transcribe() unchanged.
WHISPER_KEYS: Final = (
    "compute_type",
    "language",
    "task",
    "beam_size",
    "patience",
    "length_penalty",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "temperature",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
    "hallucination_silence_threshold",
    "condition_on_previous_text",
    "initial_prompt",
    "prefix",
    "hotwords",
    "suppress_blank",
    "suppress_tokens",
    "without_timestamps",
    "max_initial_timestamp",
    "word_timestamps",
    "multilingual",
    "vad_filter",
    "max_new_tokens",
    "chunk_length",
    "clip_timestamps",
)
# The values fixed by the protocol, each with the reason given when a block differs:
# English, one temperature of 0, no earlier or prompt text, and no rule that drops or
# re-decodes the text of a window. They also keep every argument in WHISPER_INERT inert.
_FIXED: Final[Mapping[str, tuple[object, str]]] = {
    "language": ("en", "every clip is read as English"),
    "task": ("transcribe", "every clip is transcribed, never translated"),
    "multilingual": (False, "no window detects its language again"),
    "temperature": (0.0, "one temperature: one beam decode per window, never random sampling"),
    "condition_on_previous_text": (False, "no window sees the text of an earlier one"),
    "initial_prompt": (None, "no text in the prompt"),
    "prefix": (None, "no text in the prompt"),
    "hotwords": (None, "no text in the prompt"),
    "compression_ratio_threshold": (None, "no rule re-decodes a window"),
    "log_prob_threshold": (None, "no rule re-decodes or drops a window"),
    "no_speech_threshold": (None, "no window is dropped as silent"),
    "hallucination_silence_threshold": (None, "no stretch is skipped as a hallucination"),
    "vad_filter": (False, "no voice-activity filter cuts the audio"),
    "word_timestamps": (False, "no word times, which the hallucination filter needs"),
    "clip_timestamps": ("0", "the whole clip is read"),
}
WHISPER_FIXED: Final[Mapping[str, object]] = MappingProxyType(
    {key: value for key, (value, _) in _FIXED.items()}
)


def _whole(value: object) -> bool:
    """True for a whole number; True and False are not numbers here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> bool:
    """True for a whole or decimal number; True and False are not numbers here."""
    return _whole(value) or isinstance(value, float)


_KINDS: Final[Mapping[str, Callable[[object], bool]]] = {
    "text": lambda value: isinstance(value, str),
    "true or false": lambda value: isinstance(value, bool),
    "a whole number": _whole,
    "a number": _number,
    "a whole number or null": lambda value: value is None or _whole(value),
    "a list of whole numbers or null": lambda value: (
        value is None or (isinstance(value, list) and all(_whole(item) for item in value))
    ),
}
# The keys whose value the protocol leaves to the config, by the kind of value they take.
_WHISPER_KINDS: Final[Mapping[str, str]] = {
    "compute_type": "text",
    "beam_size": "a whole number",
    "patience": "a number",
    "length_penalty": "a number",
    "repetition_penalty": "a number",
    "no_repeat_ngram_size": "a whole number",
    "suppress_blank": "true or false",
    "suppress_tokens": "a list of whole numbers or null",
    "without_timestamps": "true or false",
    "max_initial_timestamp": "a number",
    "max_new_tokens": "a whole number or null",
    "chunk_length": "a whole number or null",
}
# The arguments of transcribe() that cannot change the text with these settings; they
# keep faster-whisper's defaults. best_of is used only when sampling (a temperature above
# 0); prompt_reset_on_temperature only with earlier text; prepend_punctuations and
# append_punctuations only with word times; vad_parameters only with the voice-activity
# filter; language_detection_threshold and language_detection_segments only without a
# fixed language; log_progress draws a progress bar.
WHISPER_INERT: Final = frozenset(
    {
        "best_of",
        "prompt_reset_on_temperature",
        "prepend_punctuations",
        "append_punctuations",
        "vad_parameters",
        "language_detection_threshold",
        "language_detection_segments",
        "log_progress",
    }
)
# What a reading records about one clip, never text.
WHISPER_DIAGNOSTICS: Final = (
    "windows",
    "compression_ratios",
    "lowest_mean_log_prob",
    "highest_no_speech_prob",
    "default_dropped_windows",
    "default_dropped_words",
)
WAV2VEC2_DIAGNOSTICS: Final = ("frames", "special_symbols")

# wav2vec2 ----------------------------------------------------------------------------

CTC_BLANK: Final = "<pad>"  # the CTC blank of the pinned model: its pad symbol, id 0
WORD_DELIMITER: Final = "|"
CTC_SPECIAL: Final = ("<s>", "</s>", "<unk>")  # dropped and counted, never read as letters


# Readings ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reading:
    """What a recogniser read in one clip.

    The text is participant text: it stays in the run folder on OMNI, and the repr
    leaves it out, so that no error message or log line can carry it.

    Attributes:
        text: The words read, joined by single spaces, before ``wordcheck.words``.
        seconds: Wall time of the reading, decoding included.
        diagnostics: Numbers about the clip, never text: for Whisper the keys of
            ``WHISPER_DIAGNOSTICS``, for wav2vec2 those of ``WAV2VEC2_DIAGNOSTICS``.
    """

    text: str = field(repr=False)
    seconds: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class Recogniser(Protocol):
    """What the word check needs of a recogniser; the tests pass stand-ins."""

    @property
    def name(self) -> str:
        """Short name in the tables: ``whisper`` or ``wav2vec2``."""
        ...

    def read(self, audio: np.ndarray) -> Reading:
        """Read one clip of float32 samples at 16 kHz within [-1, 1], shape (samples,)."""
        ...


def check_audio(audio: np.ndarray) -> None:
    """Refuse audio that the recognisers must not read.

    Both read a clip's stored 16-bit samples divided by 32,768: float32, one channel,
    every value within [-1, 1]. faster-whisper uses an array as it is, so an int16 array
    would reach it as values up to 32,767.

    Args:
        audio: One clip, shape (samples,).

    Raises:
        ValueError: If it is not a float32 array of shape (samples,) with at least one
            sample, or a value lies outside [-1, 1] or is not a number.
    """
    if audio.dtype != np.float32:
        raise ValueError(f"the recognisers read float32 samples, not {audio.dtype}")
    if audio.ndim != 1:
        raise ValueError(f"the recognisers read one channel, shape (samples,), not {audio.shape}")
    if audio.size == 0:
        raise ValueError("a clip must hold at least one sample")
    if not bool(np.all(np.abs(audio) <= 1.0)):
        raise ValueError("the recognisers read samples within [-1, 1]")


# Whisper -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    """Whisper's settings, from ``configs/pilot/data.yaml`` (``word_check.whisper``).

    Attributes:
        compute_type: CTranslate2's compute type, given to ``WhisperModel``.
        options: Every other key of the block, given to ``transcribe()`` unchanged.
    """

    compute_type: str
    options: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "WhisperSettings":
        """Read and check the whisper block of the whole data config.

        The block must name every key of ``WHISPER_KEYS`` and no other, each with a value
        of its kind, and hold the values fixed by the protocol (``WHISPER_FIXED``):
        English, one temperature of 0, no earlier or prompt text, and every rule that
        drops or re-decodes the text of a window switched off.

        Args:
            config: The data config as loaded from YAML, with ``word_check.whisper``.

        Returns:
            The settings.

        Raises:
            ValueError: If the block is missing, has an unknown or a missing key, or holds
                a value of the wrong kind or other than the fixed one.
        """
        word_check = config.get("word_check")
        block = word_check.get("whisper") if isinstance(word_check, Mapping) else None
        if not isinstance(block, Mapping):
            raise ValueError("the data config has no word_check.whisper block")
        unknown = sorted(str(key) for key in block if key not in WHISPER_KEYS)
        if unknown:
            raise ValueError(f"word_check.whisper has unknown keys: {', '.join(unknown)}")
        missing = [key for key in WHISPER_KEYS if key not in block]
        if missing:
            raise ValueError(f"word_check.whisper lacks the keys {', '.join(missing)}")
        for key, (expected, reason) in _FIXED.items():
            value = block[key]
            if type(value) is not type(expected) or value != expected:
                raise ValueError(
                    f"word_check.whisper.{key} must be {json.dumps(expected)} ({reason}),"
                    f" not {value!r}"
                )
        for key, kind in _WHISPER_KINDS.items():
            if not _KINDS[kind](block[key]):
                raise ValueError(f"word_check.whisper.{key} must be {kind}, not {block[key]!r}")
        options = {key: copy.deepcopy(block[key]) for key in WHISPER_KEYS[1:]}
        return cls(compute_type=block["compute_type"], options=MappingProxyType(options))

    def transcribe_options(self) -> dict[str, Any]:
        """Return the arguments of ``transcribe()`` as a new copy for each call.

        faster-whisper 1.2.1 extends in place a list of suppressed tokens that does not
        hold -1, so each call gets its own copies, and the settings never change.

        Returns:
            Every key of the block but ``compute_type``, with its value.
        """
        return copy.deepcopy(dict(self.options))


def _dropped_by_defaults(segment: Any) -> bool:
    """Whether faster-whisper 1.2.1's default silence rule would drop this segment's window."""
    skip = float(segment.no_speech_prob) > DEFAULT_NO_SPEECH_THRESHOLD
    if float(segment.avg_logprob) > DEFAULT_LOG_PROB_THRESHOLD:
        skip = False  # as transcribe.py: a window read with enough confidence stays
    return skip


def _window_diagnostics(segments: Sequence[Any], texts: Sequence[str]) -> dict[str, Any]:
    """Numbers about the windows of one reading, never text.

    Every segment of a window carries the window's first frame (``seek``), mean
    log-probability, no-speech probability and compression ratio. A window that gave no
    segment holds no words and is not counted.

    Args:
        segments: The segments of one reading, in order.
        texts: Their stripped texts.

    Returns:
        The keys of ``WHISPER_DIAGNOSTICS``. The lowest mean log-probability and the
        highest no-speech probability are NaN when no window gave a segment.
    """
    windows: dict[int, list[int]] = {}
    for index, segment in enumerate(segments):
        windows.setdefault(int(segment.seek), []).append(index)
    first = [segments[indices[0]] for indices in windows.values()]
    dropped = [
        indices for indices in windows.values() if _dropped_by_defaults(segments[indices[0]])
    ]
    return {
        "windows": len(windows),
        "compression_ratios": tuple(float(segment.compression_ratio) for segment in first),
        "lowest_mean_log_prob": min(
            (float(segment.avg_logprob) for segment in first), default=math.nan
        ),
        "highest_no_speech_prob": max(
            (float(segment.no_speech_prob) for segment in first), default=math.nan
        ),
        "default_dropped_windows": len(dropped),
        "default_dropped_words": sum(
            len(texts[index].split()) for indices in dropped for index in indices
        ),
    }


class Whisper:
    """Whisper large-v3, run by faster-whisper with exactly the word check's settings.

    Build it with ``Whisper.load`` on OMNI. The constructor takes the model directly, so
    that the tests can pass a stand-in.

    Args:
        model: faster-whisper's ``WhisperModel``, or a stand-in with its ``transcribe``.
        settings: Whisper's settings.
        identity: What ``run.json`` records about the loaded model.

    Attributes:
        resolved_options: The options as faster-whisper resolved them in this object's
            first read (``asdict`` of its ``TranscriptionOptions``, which holds no text
            and no clip's duration); None before it.
    """

    name: Final = "whisper"

    def __init__(
        self, model: Any, settings: WhisperSettings, identity: Mapping[str, Any] | None = None
    ) -> None:
        self.model = model
        self.settings = settings
        self.identity = dict(identity or {})
        self.resolved_options: dict[str, Any] | None = None

    @classmethod
    def load(cls, snapshot: Path, settings: WhisperSettings, device: str = "cuda") -> "Whisper":
        """Load Whisper offline from the pinned snapshot and check it (OMNI only).

        - PyTorch is imported before faster-whisper: CTranslate2 links no CUDA library
          and loads cuBLAS by name when it first needs it, and the only copy in the
          environment comes with PyTorch's CUDA build.
        - The snapshot must hold the five files of ``WHISPER_FILES``. Without the
          tokenizer, faster-whisper would fetch one from the Hub; without the feature
          extractor's settings, it would quietly build 80 mel bands.
        - The model runs in the compute type of the settings (float16), and its feature
          extractor must give large-v3's 128 mel bands.

        Args:
            snapshot: Folder of the pinned ``Systran/faster-whisper-large-v3`` snapshot.
            settings: Whisper's settings.
            device: Where the model runs.

        Returns:
            Whisper, with ``identity`` filled in.

        Raises:
            StepError: If a file is missing from the snapshot, or the feature extractor
                does not give 128 mel bands.
        """
        import torch  # noqa: F401  # first, so that CTranslate2 finds PyTorch's cuBLAS
        from faster_whisper import WhisperModel

        missing = [name for name in WHISPER_FILES if not (snapshot / name).is_file()]
        if missing:
            raise StepError(f"the Whisper snapshot lacks {', '.join(missing)}")
        model = WhisperModel(str(snapshot), device=device, compute_type=settings.compute_type)
        bands = int(model.feature_extractor.mel_filters.shape[0])
        if bands != MEL_BANDS:
            raise StepError(f"Whisper's features have {bands} mel bands, not {MEL_BANDS}")
        identity = {"compute_type": settings.compute_type, "device": device, "mel_bands": bands}
        return cls(model, settings, identity)

    def read(self, audio: np.ndarray) -> Reading:
        """Read one clip with exactly the arguments of the whisper block.

        ``transcribe()`` returns a lazy result: the windows are decoded while its
        segments are taken, so that happens inside the timer. The text is the segments'
        stripped texts, in order, joined by single spaces.

        Args:
            audio: One clip, float32 at 16 kHz within [-1, 1], shape (samples,).

        Returns:
            The reading, with the diagnostics of ``WHISPER_DIAGNOSTICS``: the windows that
            gave text, each one's compression ratio, the lowest mean log-probability, the
            highest no-speech probability, and the windows and words that faster-whisper's
            default silence rule would have dropped.

        Raises:
            ValueError: If the audio is refused (``check_audio``).
        """
        check_audio(audio)
        options = self.settings.transcribe_options()
        start = perf_counter()
        lazy, info = self.model.transcribe(audio, **options)
        segments = list(lazy)
        seconds = perf_counter() - start
        if self.resolved_options is None:
            self.resolved_options = asdict(info.transcription_options)
        texts = [str(segment.text).strip() for segment in segments]
        return Reading(
            text=" ".join(text for text in texts if text),
            seconds=seconds,
            diagnostics=_window_diagnostics(segments, texts),
        )


# wav2vec2 ----------------------------------------------------------------------------


def greedy_ctc(frame_ids: Iterable[int], vocab: Mapping[str, int]) -> tuple[str, int]:
    """Decode the best symbol of each frame greedily, as a CTC model's reading.

    Repeats are merged first, so a letter said twice stays twice when a blank lies
    between them; then the blank is dropped, and ``<s>``, ``</s>`` and ``<unk>`` are
    dropped and counted (a run of one counts once). A run of word delimiters becomes one
    space, and none is kept at either end. The tokenizer's own decoding would read
    ``<unk>`` and ``<s>`` as text, and with ``skip_special_tokens=True`` it drops the
    blank before merging, which merges real double letters.

    Args:
        frame_ids: The id of the most likely symbol in each frame, in order.
        vocab: The model's symbols and their ids, with the blank and the word delimiter.

    Returns:
        The text, words joined by single spaces, and the count of special symbols dropped.

    Raises:
        ValueError: If two symbols share an id, the vocabulary lacks the blank or the word
            delimiter, or a frame's id is not in it.
    """
    symbols = {index: symbol for symbol, index in vocab.items()}
    if len(symbols) != len(vocab):
        raise ValueError("two symbols of the vocabulary share an id")
    if CTC_BLANK not in vocab or WORD_DELIMITER not in vocab:
        raise ValueError("the vocabulary needs the CTC blank and the word delimiter")
    pieces: list[str] = []
    special = 0
    for index, _ in itertools.groupby(frame_ids):
        symbol = symbols.get(int(index))
        if symbol is None:
            raise ValueError(f"a frame's id {int(index)} is not among the {len(vocab)} symbols")
        if symbol == CTC_BLANK:
            continue
        if symbol in CTC_SPECIAL:
            special += 1
            continue
        pieces.append(" " if symbol == WORD_DELIMITER else symbol)
    return " ".join("".join(pieces).split()), special


def check_float32_eval(model: Any) -> None:
    """Stop unless every weight of a PyTorch model is float32 and all of it is in evaluation mode.

    In training mode, wav2vec2's dropout, LayerDrop and time masking would act.

    Args:
        model: A PyTorch module.

    Raises:
        StepError: If a floating-point weight or buffer is not float32, or any part of the
            model is in training mode.
    """
    import torch

    tensors = [*model.parameters(), *model.buffers()]
    other = sorted(
        {str(t.dtype) for t in tensors if t.is_floating_point() and t.dtype != torch.float32}
    )
    if other:
        raise StepError(f"the model must run in float32, but it holds {', '.join(other)}")
    if any(module.training for module in model.modules()):
        raise StepError("the model is in training mode, so its dropout would be on")


class Wav2vec2:
    """The pinned wav2vec2 CTC model in float32, read greedily, one clip at a time.

    Build it with ``Wav2vec2.load``. The constructor takes the parts directly.

    Args:
        model: The loaded ``Wav2Vec2ForCTC``.
        processor: Its ``Wav2Vec2Processor``.
        device: Where the model runs.
        identity: What ``run.json`` records about the loaded model.
    """

    name: Final = "wav2vec2"

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str = "cuda",
        identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.vocab: dict[str, int] = dict(processor.tokenizer.get_vocab())
        self.identity = dict(identity or {})

    @classmethod
    def load(cls, snapshot: Path, device: str = "cuda") -> "Wav2vec2":
        """Load wav2vec2 offline from the pinned snapshot in float32 and check it.

        The weights load as float32, the model's own precision, and every part of the
        model must be float32 and in evaluation mode (``check_float32_eval``). The
        vocabulary must hold the blank, the word delimiter and the special symbols, the
        blank must be the model's pad symbol, and the model must score every symbol.

        Args:
            snapshot: Folder of the pinned ``facebook/wav2vec2-large-960h-lv60-self``
                snapshot.
            device: Where the model runs.

        Returns:
            wav2vec2, with ``identity`` filled in.

        Raises:
            StepError: If a check fails.
        """
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(snapshot, local_files_only=True)
        loaded: Any = Wav2Vec2ForCTC.from_pretrained(
            snapshot, dtype=torch.float32, local_files_only=True
        )
        model = loaded.to(device).eval()
        check_float32_eval(model)
        vocab = dict(processor.tokenizer.get_vocab())
        lacking = [
            symbol for symbol in (CTC_BLANK, WORD_DELIMITER, *CTC_SPECIAL) if symbol not in vocab
        ]
        if lacking:
            raise StepError(f"wav2vec2's vocabulary lacks {len(lacking)} of its CTC symbols")
        if vocab[CTC_BLANK] != model.config.pad_token_id:
            raise StepError("wav2vec2's CTC blank is not its pad symbol")
        if model.config.vocab_size != len(vocab):
            raise StepError(
                f"wav2vec2 scores {model.config.vocab_size} symbols, its vocabulary has {len(vocab)}"
            )
        identity = {
            "dtype": "float32",
            "device": device,
            "attention": getattr(model.config, "_attn_implementation", None),
            "symbols": len(vocab),
        }
        return cls(model, processor, device, identity)

    def logits(self, audio: np.ndarray) -> np.ndarray:
        """Score every symbol in every 20 ms frame of one clip.

        The processor scales each clip to zero mean and unit variance, so its level does
        not matter. One clip goes in alone and unpadded, so no attention mask is needed.

        Args:
            audio: One clip, float32 at 16 kHz within [-1, 1], shape (samples,).

        Returns:
            The scores, shape (frames, symbols), as float32.

        Raises:
            ValueError: If the audio is refused (``check_audio``).
        """
        import torch

        check_audio(audio)
        deterministic_torch()
        inputs = self.processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        with torch.inference_mode():
            scores = self.model(inputs.input_values.to(self.device)).logits[0]
        return np.asarray(scores.cpu().numpy(), dtype=np.float32)

    def read(self, audio: np.ndarray) -> Reading:
        """Read one clip: the most likely symbol in each frame, decoded by ``greedy_ctc``.

        Args:
            audio: One clip, float32 at 16 kHz within [-1, 1], shape (samples,).

        Returns:
            The reading in capitals, the apostrophe and spaces, with the diagnostics of
            ``WAV2VEC2_DIAGNOSTICS``: the frames and the special symbols dropped.

        Raises:
            ValueError: If the audio is refused (``check_audio``).
        """
        start = perf_counter()
        scores = self.logits(audio)
        text, special = greedy_ctc(np.argmax(scores, axis=-1).tolist(), self.vocab)
        seconds = perf_counter() - start
        return Reading(
            text=text,
            seconds=seconds,
            diagnostics={"frames": int(scores.shape[0]), "special_symbols": special},
        )
