import ast
import copy
import inspect
import json
import logging
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml

from listening_beyond_words import recognisers as rec
from listening_beyond_words.runinfo import StepError

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "pilot" / "data.yaml"
RATE = 16_000
# The whisper block of the plan (notes, plan of step 6.7, section "Settings").
PLAN_WHISPER: dict[str, Any] = {
    "compute_type": "float16",
    "language": "en",
    "task": "transcribe",
    "beam_size": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "temperature": 0.0,
    "compression_ratio_threshold": None,
    "log_prob_threshold": None,
    "no_speech_threshold": None,
    "hallucination_silence_threshold": None,
    "condition_on_previous_text": False,
    "initial_prompt": None,
    "prefix": None,
    "hotwords": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": False,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False,
    "multilingual": False,
    "vad_filter": False,
    "max_new_tokens": None,
    "chunk_length": None,
    "clip_timestamps": "0",
}
# The layout of the pinned wav2vec2 vocabulary: blank, special symbols, word delimiter,
# then the capitals and the apostrophe.
VOCAB = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "|": 4,
    **{symbol: 5 + index for index, symbol in enumerate("ETAONIHSRDLUMWCFGYPBVK'XJQZ")},
}
HEAVY = ("torch", "transformers", "faster_whisper", "ctranslate2", "opensmile")


@pytest.fixture
def block() -> dict[str, Any]:
    """The plan's whisper block, a fresh copy for each test."""
    return copy.deepcopy(PLAN_WHISPER)


def config_with(block: dict[str, Any]) -> dict[str, Any]:
    return {"word_check": {"whisper": block}}


def plan_options() -> dict[str, Any]:
    return {key: value for key, value in PLAN_WHISPER.items() if key != "compute_type"}


def tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# The whisper block -------------------------------------------------------------------


def test_the_plans_block_loads_with_one_temperature_english_and_no_rule_that_drops_text(
    block: dict[str, Any],
) -> None:
    settings = rec.WhisperSettings.from_config(config_with(block))
    options = settings.options
    assert settings.compute_type == "float16"
    assert options["temperature"] == 0.0
    assert isinstance(options["temperature"], float)  # one value, not a list
    assert (options["language"], options["task"], options["multilingual"]) == (
        "en",
        "transcribe",
        False,
    )
    for rule in (
        "no_speech_threshold",
        "log_prob_threshold",
        "compression_ratio_threshold",
        "hallucination_silence_threshold",
    ):
        assert options[rule] is None
    assert options["vad_filter"] is False
    assert options["word_timestamps"] is False
    assert options["condition_on_previous_text"] is False
    assert settings.transcribe_options() == plan_options()


def test_the_whisper_block_of_data_yaml_passes_the_check_with_the_plans_values(
    block: dict[str, Any],
) -> None:
    config = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    settings = rec.WhisperSettings.from_config(config)
    assert settings.compute_type == block["compute_type"]
    assert settings.transcribe_options() == {
        key: value for key, value in block.items() if key != "compute_type"
    }


def test_the_block_refuses_an_unknown_key(block: dict[str, Any]) -> None:
    block["best_of"] = 5  # inert with one temperature: left at faster-whisper's default
    with pytest.raises(ValueError, match="unknown keys: best_of"):
        rec.WhisperSettings.from_config(config_with(block))


@pytest.mark.parametrize("key", rec.WHISPER_KEYS)
def test_the_block_refuses_a_missing_key(block: dict[str, Any], key: str) -> None:
    del block[key]
    with pytest.raises(ValueError, match=f"lacks the keys {key}$"):
        rec.WhisperSettings.from_config(config_with(block))


@pytest.mark.parametrize("config", [{}, {"word_check": {}}, {"word_check": {"whisper": [1]}}])
def test_a_config_without_a_whisper_block_is_refused(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"no word_check\.whisper block"):
        rec.WhisperSettings.from_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("temperature", [0.0]),
        ("temperature", 0.2),
        ("temperature", 0),
        ("language", None),
        ("language", "de"),
        ("task", "translate"),
        ("multilingual", True),
        ("no_speech_threshold", 0.6),
        ("log_prob_threshold", -1.0),
        ("compression_ratio_threshold", 2.4),
        ("hallucination_silence_threshold", 2.0),
        ("vad_filter", True),
        ("vad_filter", 0),
        ("word_timestamps", True),
        ("condition_on_previous_text", True),
        ("initial_prompt", "Hello."),
        ("prefix", "So"),
        ("hotwords", "words"),
        ("clip_timestamps", "0,10"),
    ],
)
def test_the_block_refuses_every_other_value_of_a_fixed_key(
    block: dict[str, Any], key: str, value: object
) -> None:
    block[key] = value
    with pytest.raises(ValueError, match=f"word_check.whisper.{key} must be"):
        rec.WhisperSettings.from_config(config_with(block))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("compute_type", 16),
        ("beam_size", "5"),
        ("beam_size", True),
        ("beam_size", 5.0),
        ("patience", "1.0"),
        ("patience", False),
        ("suppress_blank", 1),
        ("suppress_tokens", [-1.0]),
        ("suppress_tokens", "-1"),
        ("without_timestamps", "false"),
        ("max_initial_timestamp", None),
        ("max_new_tokens", 1.5),
        ("chunk_length", "30"),
    ],
)
def test_the_block_refuses_a_value_of_the_wrong_kind(
    block: dict[str, Any], key: str, value: object
) -> None:
    block[key] = value
    with pytest.raises(ValueError, match=f"word_check.whisper.{key} must be"):
        rec.WhisperSettings.from_config(config_with(block))


@pytest.mark.parametrize("key", [key for key in rec.WHISPER_KEYS if key not in rec.WHISPER_FIXED])
def test_every_key_that_is_not_fixed_has_its_kind_checked(block: dict[str, Any], key: str) -> None:
    block[key] = object()
    with pytest.raises(ValueError, match=f"word_check.whisper.{key} must be"):
        rec.WhisperSettings.from_config(config_with(block))


def test_the_block_leaves_the_other_values_to_the_config(block: dict[str, Any]) -> None:
    block.update(beam_size=1, suppress_tokens=None, max_new_tokens=100, patience=2)
    options = rec.WhisperSettings.from_config(config_with(block)).transcribe_options()
    assert (options["beam_size"], options["suppress_tokens"], options["max_new_tokens"]) == (
        1,
        None,
        100,
    )


def test_the_keys_and_the_inert_arguments_are_named_once() -> None:
    assert len(rec.WHISPER_KEYS) == len(set(rec.WHISPER_KEYS)) == 27
    assert set(rec.WHISPER_FIXED) < set(rec.WHISPER_KEYS)
    assert len(rec.WHISPER_INERT) == 8
    assert rec.WHISPER_INERT.isdisjoint(rec.WHISPER_KEYS)


def test_each_call_gets_its_own_copy_of_the_settings(block: dict[str, Any]) -> None:
    settings = rec.WhisperSettings.from_config(config_with(block))
    block["suppress_tokens"].append(7)  # the config read once, changed later
    first = settings.transcribe_options()
    first["suppress_tokens"].append(5)  # as faster-whisper may extend the list it is given
    assert settings.transcribe_options()["suppress_tokens"] == [-1]
    assert "compute_type" not in first


# Whisper's reader --------------------------------------------------------------------


@dataclass
class FakeSegment:
    """What faster-whisper yields: one segment, with the numbers of its window."""

    text: str
    seek: int = 0
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.01
    compression_ratio: float = 1.5


@dataclass
class FakeOptions:
    """Stands in for faster-whisper's TranscriptionOptions (a dataclass)."""

    beam_size: int
    temperatures: list[float]
    call: int


class FakeClock:
    """A clock that moves only when the fake model decodes."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeWhisperModel:
    """Stands in for faster-whisper's WhisperModel: records each call, returns a lazy result."""

    def __init__(self, segments: list[FakeSegment], clock: FakeClock | None = None) -> None:
        self.segments = segments
        self.clock = clock
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[Any, Any]:
        self.calls.append((audio, options))
        resolved = FakeOptions(options["beam_size"], [options["temperature"]], len(self.calls))
        info = SimpleNamespace(transcription_options=resolved, duration=len(audio) / RATE)

        def lazy() -> Any:
            for segment in self.segments:
                if self.clock is not None:
                    self.clock.now += 5.0  # decoding happens while the segments are taken
                yield segment

        return lazy(), info


def whisper_with(segments: list[FakeSegment], clock: FakeClock | None = None) -> rec.Whisper:
    settings = rec.WhisperSettings.from_config(config_with(copy.deepcopy(PLAN_WHISPER)))
    return rec.Whisper(FakeWhisperModel(segments, clock), settings)


def test_whisper_passes_the_array_and_exactly_the_settings() -> None:
    whisper = whisper_with([FakeSegment(" One.")])
    clip = tone()
    whisper.read(clip)
    audio, options = whisper.model.calls[0]
    assert audio is clip
    assert options == plan_options()
    assert {key: type(value) for key, value in options.items()} == {
        key: type(value) for key, value in plan_options().items()
    }


def test_whisper_joins_the_stripped_texts_in_order_and_counts_windows() -> None:
    whisper = whisper_with(
        [
            FakeSegment(" Hello there.", seek=0, compression_ratio=1.2),
            FakeSegment("  how are you ", seek=0, compression_ratio=1.2),
            FakeSegment(" ", seek=0, compression_ratio=1.2),
            FakeSegment(" Second window. ", seek=2800, compression_ratio=2.5),
        ]
    )
    reading = whisper.read(tone())
    assert reading.text == "Hello there. how are you Second window."
    assert reading.diagnostics["windows"] == 2
    assert reading.diagnostics["compression_ratios"] == (1.2, 2.5)
    assert tuple(reading.diagnostics) == rec.WHISPER_DIAGNOSTICS


def test_whisper_counts_the_windows_and_words_that_the_default_silence_rule_would_drop() -> None:
    whisper = whisper_with(
        [
            FakeSegment(" a b", seek=0, avg_logprob=-2.0, no_speech_prob=0.9),
            FakeSegment(" c", seek=0, avg_logprob=-2.0, no_speech_prob=0.9),
            FakeSegment(" d e f", seek=2800, avg_logprob=-0.5, no_speech_prob=0.9),
            FakeSegment(" g", seek=5600, avg_logprob=-2.0, no_speech_prob=0.3),
        ]
    )
    reading = whisper.read(tone())
    assert reading.text.split() == ["a", "b", "c", "d", "e", "f", "g"]  # every word kept
    numbers = reading.diagnostics
    assert numbers["windows"] == 3
    assert numbers["default_dropped_windows"] == 1
    assert numbers["default_dropped_words"] == 3
    assert numbers["lowest_mean_log_prob"] == -2.0
    assert numbers["highest_no_speech_prob"] == 0.9


@pytest.mark.parametrize(
    ("no_speech", "mean_log_prob", "dropped"),
    [(0.6, -2.0, 0), (0.61, -1.0, 1), (0.9, -0.99, 0), (1.0, -5.0, 1)],
)
def test_the_default_silence_rule_is_read_as_faster_whisper_writes_it(
    no_speech: float, mean_log_prob: float, dropped: int
) -> None:
    segment = FakeSegment(" w", avg_logprob=mean_log_prob, no_speech_prob=no_speech)
    reading = whisper_with([segment]).read(tone())
    assert reading.diagnostics["default_dropped_windows"] == dropped
    assert reading.diagnostics["default_dropped_words"] == dropped


def test_whisper_with_no_segment_reads_nothing() -> None:
    reading = whisper_with([]).read(tone())
    assert reading.text == ""
    numbers = reading.diagnostics
    assert (numbers["windows"], numbers["compression_ratios"]) == (0, ())
    assert np.isnan(numbers["lowest_mean_log_prob"])
    assert np.isnan(numbers["highest_no_speech_prob"])
    assert (numbers["default_dropped_windows"], numbers["default_dropped_words"]) == (0, 0)


def test_whisper_consumes_the_lazy_result_inside_the_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(rec, "perf_counter", clock)
    whisper = whisper_with([FakeSegment(" one"), FakeSegment(" two", seek=2800)], clock)
    assert whisper.read(tone()).seconds == pytest.approx(10.0)


def test_whisper_keeps_the_options_of_its_first_read() -> None:
    whisper = whisper_with([FakeSegment(" one")])
    assert whisper.resolved_options is None
    whisper.read(tone())
    whisper.read(tone())
    assert whisper.resolved_options == {"beam_size": 5, "temperatures": [0.0], "call": 1}


def test_a_readings_repr_leaves_out_its_text() -> None:
    reading = whisper_with([FakeSegment(" SYNTHETICWORD")]).read(tone())
    assert reading.text == "SYNTHETICWORD"
    assert "SYNTHETICWORD" not in repr(reading)
    assert "seconds" in repr(reading)


def test_the_recognisers_name_themselves() -> None:
    assert (rec.Whisper.name, rec.Wav2vec2.name) == ("whisper", "wav2vec2")


# Whisper's loader (a stand-in for faster-whisper) --------------------------------------


@pytest.fixture
def fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A faster_whisper module whose WhisperModel records how it was made."""

    class FakeModel:
        bands = 128
        made: ClassVar[list[tuple[str, dict[str, Any]]]] = []

        def __init__(self, path: str, **kwargs: Any) -> None:
            FakeModel.made.append((path, kwargs))
            filters = np.zeros((FakeModel.bands, 201), dtype=np.float32)
            self.feature_extractor = SimpleNamespace(mel_filters=filters)

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return FakeModel


def whisper_snapshot(folder: Path, files: tuple[str, ...] = rec.WHISPER_FILES) -> Path:
    folder.mkdir()
    for name in files:
        (folder / name).write_bytes(b"")
    return folder


def test_whisper_loads_in_the_compute_type_of_the_settings(
    fake_faster_whisper: Any, tmp_path: Path, block: dict[str, Any]
) -> None:
    snapshot = whisper_snapshot(tmp_path / "snapshot")
    settings = rec.WhisperSettings.from_config(config_with(block))
    whisper = rec.Whisper.load(snapshot, settings, device="cpu")
    assert fake_faster_whisper.made == [
        (str(snapshot), {"device": "cpu", "compute_type": "float16"})
    ]
    assert whisper.settings is settings
    assert whisper.identity == {"compute_type": "float16", "device": "cpu", "mel_bands": 128}


# Named here, not taken from the module, so that a file dropped from its list shows.
SNAPSHOT_FILES = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
    "vocabulary.json",
)


@pytest.mark.parametrize("missing", SNAPSHOT_FILES)
def test_whisper_refuses_a_snapshot_without_one_of_its_files(
    fake_faster_whisper: Any, tmp_path: Path, block: dict[str, Any], missing: str
) -> None:
    files = tuple(name for name in SNAPSHOT_FILES if name != missing)
    snapshot = whisper_snapshot(tmp_path / "snapshot", files)
    settings = rec.WhisperSettings.from_config(config_with(block))
    with pytest.raises(StepError, match=f"lacks {re.escape(missing)}$"):
        rec.Whisper.load(snapshot, settings, device="cpu")
    assert fake_faster_whisper.made == []  # refused before anything loads


def test_whisper_refuses_features_without_128_mel_bands(
    fake_faster_whisper: Any, tmp_path: Path, block: dict[str, Any]
) -> None:
    fake_faster_whisper.bands = 80  # what faster-whisper builds without the extractor's settings
    snapshot = whisper_snapshot(tmp_path / "snapshot")
    settings = rec.WhisperSettings.from_config(config_with(block))
    with pytest.raises(StepError, match="80 mel bands, not 128"):
        rec.Whisper.load(snapshot, settings, device="cpu")


# The input guard ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(160, dtype=np.int16),
        np.zeros(160, dtype=np.float64),
        np.zeros((2, 160), dtype=np.float32),
        np.zeros((160, 2), dtype=np.float32),
        np.array([0.5, 1.5], dtype=np.float32),
        np.array([-1.0001, 0.0], dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([], dtype=np.float32),
    ],
    ids=[
        "int16",
        "float64",
        "two-rows",
        "two-columns",
        "above-1",
        "below-1",
        "nan",
        "inf",
        "empty",
    ],
)
def test_the_input_guard_refuses_audio_the_recognisers_must_not_read(audio: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"recognisers read|at least one sample"):
        rec.check_audio(audio)


def test_the_input_guard_takes_stored_samples_divided_by_32768() -> None:
    samples = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    rec.check_audio((samples / 32_768).astype(np.float32))
    rec.check_audio(np.array([-1.0, 1.0], dtype=np.float32))


def test_whisper_refuses_bad_audio_before_reading() -> None:
    whisper = whisper_with([FakeSegment(" one")])
    with pytest.raises(ValueError, match="int16"):
        whisper.read(np.zeros(RATE, dtype=np.int16))
    assert whisper.model.calls == []


# Greedy CTC decoding -----------------------------------------------------------------


def frames(spec: str) -> list[int]:
    """Frame ids from symbols separated by spaces, such as "H E L <pad> L O"."""
    return [VOCAB[symbol] for symbol in spec.split()]


def test_greedy_ctc_keeps_a_double_letter_with_a_blank_between_and_merges_repeats() -> None:
    assert rec.greedy_ctc(frames("H H E L <pad> L L O"), VOCAB) == ("HELLO", 0)
    assert rec.greedy_ctc(frames("H E L L L O"), VOCAB) == ("HELO", 0)


def test_greedy_ctc_turns_a_run_of_delimiters_into_one_space() -> None:
    spec = "| A | | <pad> | B <pad> <pad> | C | |"
    assert rec.greedy_ctc(frames(spec), VOCAB) == ("A B C", 0)
    assert rec.greedy_ctc(frames("I | ' M"), VOCAB) == ("I 'M", 0)


def test_greedy_ctc_drops_and_counts_the_special_symbols() -> None:
    assert rec.greedy_ctc(frames("<s> H E L <unk> L O </s>"), VOCAB) == ("HELLO", 3)
    assert rec.greedy_ctc(frames("A <unk> <unk> B"), VOCAB) == ("AB", 1)  # one run, one symbol
    assert rec.greedy_ctc(frames("A <unk> <pad> <unk> | B"), VOCAB) == ("A B", 2)


def test_greedy_ctc_gives_nothing_for_blanks_and_delimiters_only() -> None:
    assert rec.greedy_ctc(frames("<pad> <pad> <pad>"), VOCAB) == ("", 0)
    assert rec.greedy_ctc(frames("| <pad> |"), VOCAB) == ("", 0)
    assert rec.greedy_ctc([], VOCAB) == ("", 0)


def test_greedy_ctc_takes_numpy_ids() -> None:
    assert rec.greedy_ctc(np.array(frames("O K")), VOCAB) == ("OK", 0)


def test_greedy_ctc_refuses_an_unknown_id_and_an_incomplete_vocabulary() -> None:
    with pytest.raises(ValueError, match="not among the 32 symbols"):
        rec.greedy_ctc([5, 32], VOCAB)
    without_blank = {symbol: index for symbol, index in VOCAB.items() if symbol != "<pad>"}
    with pytest.raises(ValueError, match="blank"):
        rec.greedy_ctc([5], without_blank)
    with pytest.raises(ValueError, match="share an id"):
        rec.greedy_ctc([5], {**VOCAB, "Ä": 5})


def ctc_tokenizer(folder: Path) -> Any:
    from transformers import Wav2Vec2CTCTokenizer

    path = folder / "vocab.json"
    path.write_text(json.dumps(VOCAB), encoding="utf-8")
    return Wav2Vec2CTCTokenizer(
        str(path),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
    )


def test_greedy_ctc_reads_the_tokenizers_words_and_keeps_its_double_letters(
    tmp_path: Path,
) -> None:
    tokenizer = ctc_tokenizer(tmp_path)
    # The tokenizer's own decoding can leave two spaces, and lets special symbols through.
    assert tokenizer.decode(frames("A | <pad> | B | C")) == "A  B C"
    assert "<unk>" in tokenizer.decode(frames("H E L <unk> L O"))
    plain = [index for symbol, index in VOCAB.items() if symbol not in rec.CTC_SPECIAL]
    rng = np.random.default_rng(9001)
    doubled = 0
    for _ in range(400):
        frame_ids = [int(index) for index in rng.choice(plain, size=int(rng.integers(0, 80)))]
        text, special = rec.greedy_ctc(frame_ids, VOCAB)
        assert special == 0
        assert text.split() == tokenizer.decode(frame_ids).split()
        # skip_special_tokens=True drops the blank before merging repeats, so it merges
        # real double letters, and differs exactly there.
        merged = [re.sub(r"(.)\1+", r"\1", word) for word in text.split()]
        assert tokenizer.decode(frame_ids, skip_special_tokens=True).split() == merged
        doubled += merged != text.split()
    assert doubled > 0


# wav2vec2 on a tiny random model -------------------------------------------------------


def save_tiny_wav2vec2(folder: Path, half: bool = False, **overrides: Any) -> Path:
    """A tiny random Wav2Vec2ForCTC with the pinned model's vocabulary and processor."""
    import torch
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    folder.mkdir(parents=True)
    settings: dict[str, Any] = {
        "vocab_size": 32,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "conv_dim": (16,) * 7,  # the real model's kernels and strides: one frame per 20 ms
        "feat_extract_norm": "layer",
        "do_stable_layer_norm": True,
        "num_conv_pos_embeddings": 16,
        "num_conv_pos_embedding_groups": 2,
        "pad_token_id": 0,
        **overrides,
    }
    with torch.random.fork_rng():
        torch.manual_seed(9001)
        model = Wav2Vec2ForCTC(Wav2Vec2Config(**settings))
    (model.half() if half else model).save_pretrained(folder)
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=extractor, tokenizer=ctc_tokenizer(folder))
    processor.save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def tiny_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tiny_wav2vec2(tmp_path_factory.mktemp("tiny") / "wav2vec2")


@pytest.fixture
def restore_torch_flags() -> Any:
    """Put PyTorch's determinism flags back after a test, which reading sets for the process."""
    import torch

    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    yield
    torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    torch.backends.cudnn.deterministic = saved[2]
    torch.backends.cudnn.benchmark = saved[3]


def noise(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(9002)
    return (0.01 * rng.standard_normal(round(seconds * RATE))).astype(np.float32)


def test_wav2vec2_loads_offline_in_float32_and_evaluation_mode(tiny_snapshot: Path) -> None:
    import torch

    wav2vec2 = rec.Wav2vec2.load(tiny_snapshot, device="cpu")
    assert {weight.dtype for weight in wav2vec2.model.parameters()} == {torch.float32}
    assert not any(module.training for module in wav2vec2.model.modules())
    assert wav2vec2.vocab == VOCAB
    assert wav2vec2.identity["dtype"] == "float32"
    assert wav2vec2.identity["symbols"] == 32


def test_wav2vec2_loads_a_half_precision_file_as_float32(tmp_path: Path) -> None:
    import torch

    snapshot = save_tiny_wav2vec2(tmp_path / "half", half=True)
    wav2vec2 = rec.Wav2vec2.load(snapshot, device="cpu")
    assert {weight.dtype for weight in wav2vec2.model.parameters()} == {torch.float32}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"pad_token_id": 1}, "blank is not its pad symbol"), ({"vocab_size": 33}, "scores 33")],
)
def test_wav2vec2_refuses_a_vocabulary_that_does_not_fit_the_model(
    tmp_path: Path, overrides: dict[str, int], message: str
) -> None:
    snapshot = save_tiny_wav2vec2(tmp_path / "other", **overrides)
    with pytest.raises(StepError, match=message):
        rec.Wav2vec2.load(snapshot, device="cpu")


def test_wav2vec2_gives_1870_frames_for_37_415_s_whatever_the_level(
    tiny_snapshot: Path, restore_torch_flags: None
) -> None:
    wav2vec2 = rec.Wav2vec2.load(tiny_snapshot, device="cpu")
    clip = noise(37.415)
    scores = wav2vec2.logits(clip)
    assert scores.shape == (1870, 32)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(wav2vec2.logits(clip * 10), scores, atol=1e-4)


def test_wav2vec2_reads_a_clip_the_same_way_twice(
    tiny_snapshot: Path, restore_torch_flags: None
) -> None:
    import torch

    wav2vec2 = rec.Wav2vec2.load(tiny_snapshot, device="cpu")
    clip = noise(3.0)
    first, second = wav2vec2.read(clip), wav2vec2.read(clip)
    assert first.text == second.text
    assert first.diagnostics == second.diagnostics
    assert tuple(first.diagnostics) == rec.WAV2VEC2_DIAGNOSTICS
    assert first.diagnostics["frames"] == 149
    assert set(first.text) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ' ")
    assert first.seconds > 0
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_wav2vec2_refuses_bad_audio_before_reading(tiny_snapshot: Path) -> None:
    wav2vec2 = rec.Wav2vec2.load(tiny_snapshot, device="cpu")
    with pytest.raises(ValueError, match="float64"):
        wav2vec2.read(np.zeros(RATE, dtype=np.float64))


def test_the_float32_and_evaluation_check_refuses_half_weights_and_training_mode() -> None:
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1)).eval()
    rec.check_float32_eval(model)
    with pytest.raises(StepError, match=r"torch\.float16"):
        rec.check_float32_eval(copy.deepcopy(model).half())
    model[1].train()  # one part left in training mode is enough to refuse
    with pytest.raises(StepError, match="training mode"):
        rec.check_float32_eval(model)


# Imports -------------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["recognisers", "wordcheck", "voicecheck"])
def test_the_word_check_modules_import_no_heavy_package_when_imported(module: str) -> None:
    code = (
        "import importlib, sys\n"
        f"importlib.import_module('listening_beyond_words.{module}')\n"
        f"print(' '.join(name for name in {HEAVY!r} if name in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split() == []


def imported(node: ast.AST) -> list[str]:
    """The top-level packages that an import statement names."""
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module.split(".")[0]]
    return []


def test_recognisers_imports_the_heavy_packages_inside_functions_only() -> None:
    tree = ast.parse(Path(rec.__file__).read_text(encoding="utf-8"))
    at_module_level = {name for node in tree.body for name in imported(node)}
    assert at_module_level.isdisjoint(HEAVY)


def test_whisper_load_imports_torch_before_faster_whisper() -> None:
    tree = ast.parse(Path(rec.__file__).read_text(encoding="utf-8"))
    whisper = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Whisper"
    )
    load = next(
        node for node in whisper.body if isinstance(node, ast.FunctionDef) and node.name == "load"
    )
    order = [
        name
        for _, name in sorted(
            (node.lineno, name) for node in ast.walk(load) for name in imported(node)
        )
    ]
    assert "torch" in order
    assert "faster_whisper" in order
    assert order.index("torch") < order.index("faster_whisper")
    everywhere = [name for node in ast.walk(tree) for name in imported(node)]
    assert everywhere.count("faster_whisper") == order.count("faster_whisper")  # only in load
    assert "ctranslate2" not in everywhere


# faster-whisper's own decoding loop with a stand-in for CTranslate2 (OMNI only) --------

SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|en|>",
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
)
TOKEN_IDS = {name: 1000 + index for index, name in enumerate(SPECIAL_TOKENS)}
FIRST_TIME = TOKEN_IDS["<|notimestamps|>"] + 1  # <|0.00|>; each step is 0.02 s


class FakeTokenizer:
    """Stands in for Whisper's tokenizers.Tokenizer: a token n below 1000 is the word wn."""

    def token_to_id(self, name: str) -> int | None:
        return TOKEN_IDS.get(name)

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        return SimpleNamespace(ids=[7])

    def decode(self, ids: list[int]) -> str:
        return "".join(f" w{token}" for token in ids)


class FakeCTranslate2:
    """Stands in for CTranslate2's Whisper: one scripted result per window, each call recorded.

    Each entry of the script is (tokens, mean log-probability, no-speech probability).
    """

    is_multilingual = True
    device = "cpu"
    device_index = (0,)

    def __init__(self, script: list[tuple[list[int], float, float]]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []

    def encode(self, features: Any, to_cpu: bool = False) -> None:
        return None

    def generate(self, encoder_output: Any, prompts: list[list[int]], **kwargs: Any) -> list[Any]:
        tokens, mean_log_prob, no_speech = self.script[min(len(self.calls), len(self.script) - 1)]
        self.calls.append({"prompt": list(prompts[0]), **kwargs})
        # faster-whisper recovers the mean log-probability from this score
        score = mean_log_prob * (len(tokens) + 1) / len(tokens) ** kwargs["length_penalty"]
        return [
            SimpleNamespace(sequences_ids=[list(tokens)], scores=[score], no_speech_prob=no_speech)
        ]


@pytest.fixture
def faster_whisper() -> Any:
    return pytest.importorskip("faster_whisper")


def stand_in_model(script: list[tuple[list[int], float, float]]) -> Any:
    """faster-whisper's WhisperModel with its decoding loop, around a stand-in for CTranslate2."""
    from faster_whisper.feature_extractor import FeatureExtractor
    from faster_whisper.transcribe import WhisperModel

    model = WhisperModel.__new__(WhisperModel)
    model.logger = logging.getLogger("faster_whisper")
    model.model = FakeCTranslate2(script)
    model.hf_tokenizer = FakeTokenizer()
    model.feat_kwargs = {"feature_size": rec.MEL_BANDS}
    model.feature_extractor = FeatureExtractor(feature_size=rec.MEL_BANDS)
    model.input_stride = 2
    model.num_samples_per_token = model.feature_extractor.hop_length * 2
    model.frames_per_second = 100
    model.tokens_per_second = 50
    model.time_precision = 0.02
    model.max_length = 448
    return model


def data_yaml_settings() -> rec.WhisperSettings:
    return rec.WhisperSettings.from_config(yaml.safe_load(DATA.read_text(encoding="utf-8")))


@pytest.mark.slow
def test_the_word_check_settings_keep_a_window_that_the_defaults_drop(faster_whisper: Any) -> None:
    window = [FIRST_TIME, 5, 6, 7, 8, FIRST_TIME + 600]  # <|0.00|> w5 w6 w7 w8 <|12.00|>
    clip = noise(14.0)
    model = stand_in_model([(window, -2.0, 0.9)])
    reading = rec.Whisper(model, data_yaml_settings()).read(clip)
    assert reading.text.split() == ["w5", "w6", "w7", "w8"]
    numbers = reading.diagnostics
    assert (numbers["windows"], numbers["default_dropped_windows"]) == (1, 1)
    assert numbers["default_dropped_words"] == 4
    assert len(model.model.calls) == 1  # one beam decode, never sampled again
    assert model.model.calls[0]["beam_size"] == 5
    assert "sampling_temperature" not in model.model.calls[0]
    defaults = stand_in_model([(window, -2.0, 0.9)])
    segments, _ = defaults.transcribe(clip, language="en")
    assert list(segments) == []  # the library's defaults drop the whole window


@pytest.mark.slow
def test_a_long_clip_takes_two_windows_and_the_second_sees_no_earlier_text(
    faster_whisper: Any,
) -> None:
    # A whole segment to 28.00 s, then an unfinished one, which the next window reads again.
    first = [FIRST_TIME, 5, 6, FIRST_TIME + 1400, FIRST_TIME + 1400, 9, 9]
    second = [FIRST_TIME, 11, 12, FIRST_TIME + 300]
    model = stand_in_model([(first, -0.3, 0.1), (second, -0.3, 0.1)])
    reading = rec.Whisper(model, data_yaml_settings()).read(noise(37.4))
    assert reading.diagnostics["windows"] == 2
    assert reading.text.split() == ["w5", "w6", "w11", "w12"]
    start = [TOKEN_IDS[name] for name in ("<|startoftranscript|>", "<|en|>", "<|transcribe|>")]
    assert [call["prompt"] for call in model.model.calls] == [start, start]
    assert reading.diagnostics["default_dropped_windows"] == 0


@pytest.mark.slow
def test_every_argument_of_transcribe_is_in_the_block_or_inert(faster_whisper: Any) -> None:
    parameters = inspect.signature(faster_whisper.WhisperModel.transcribe).parameters
    arguments = set(parameters) - {"self", "audio"}
    in_block = set(data_yaml_settings().transcribe_options())
    assert arguments == in_block | rec.WHISPER_INERT
    assert in_block.isdisjoint(rec.WHISPER_INERT)
    assert parameters["no_speech_threshold"].default == rec.DEFAULT_NO_SPEECH_THRESHOLD
    assert parameters["log_prob_threshold"].default == rec.DEFAULT_LOG_PROB_THRESHOLD
    assert "compute_type" in inspect.signature(faster_whisper.WhisperModel.__init__).parameters
