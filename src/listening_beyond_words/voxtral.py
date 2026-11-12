"""Voxtral Mini 3B: loading in fp16 and the yes/no score of a clip.

The model hears a clip followed by the fixed prompt. A clip's score is
z = logit("Yes") - logit("No") at the first answer position, so "Yes" and "No"
must each be a single token. A participant's score is the mean z over their
segments.
"""

import base64
import io
from pathlib import Path
from typing import Any, Final

import numpy as np
import soundfile as sf

PROMPT: Final = (
    "Based on this speech, does the speaker have elevated depressive symptoms? Answer Yes or No."
)
ANSWERS: Final = ("Yes", "No")
SAMPLING_RATE: Final = 16_000


def answer_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Return the token IDs of "Yes" and "No".

    Args:
        tokenizer: The model's tokenizer; it needs ``encode(text, add_special_tokens=False)``.

    Returns:
        The IDs of "Yes" and "No", in this order.

    Raises:
        ValueError: If an answer is not exactly one token, or both answers share a token.
    """
    ids: list[int] = []
    for answer in ANSWERS:
        tokens = list(tokenizer.encode(answer, add_special_tokens=False))
        if len(tokens) != 1:
            raise ValueError(f"{answer!r} must be one token, but is {len(tokens)}: {tokens}")
        ids.append(int(tokens[0]))
    yes_id, no_id = ids
    if yes_id == no_id:
        raise ValueError(f"'Yes' and 'No' share the token {yes_id}")
    return yes_id, no_id


def load_model(snapshot: Path, device: str = "cuda") -> tuple[Any, Any]:
    """Load the model in float16, in evaluation mode, together with its processor.

    Args:
        snapshot: Folder of the pinned snapshot (see ``models.local_snapshot``).
        device: Where the model runs.

    Returns:
        The model (``VoxtralForConditionalGeneration``) and its processor.
    """
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    processor = AutoProcessor.from_pretrained(snapshot)
    # Any: the type hints of transformers' wrapped .to() confuse mypy.
    model: Any = VoxtralForConditionalGeneration.from_pretrained(snapshot, dtype=torch.float16)
    model.to(device)
    model.eval()
    return model, processor


def clip_inputs(processor: Any, audio: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> Any:
    """Build the model input for one clip: the audio, then the prompt, in one user turn.

    Args:
        processor: The model's processor.
        audio: Mono audio samples in [-1, 1].
        sampling_rate: Sampling rate of ``audio``; the model expects 16 kHz.

    Returns:
        Token IDs, attention mask and log-mel features as PyTorch tensors (a ``BatchFeature``).

    Raises:
        ValueError: If the audio is not mono or not sampled at 16 kHz.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")
    if sampling_rate != SAMPLING_RATE:
        raise ValueError(f"expected {SAMPLING_RATE} Hz audio, got {sampling_rate} Hz")
    # The processor takes audio as an encoded file; base64 keeps it in memory.
    buffer = io.BytesIO()
    sf.write(buffer, audio, sampling_rate, format="WAV", subtype="FLOAT")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "base64": audio_b64},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(conversation)


def yes_no_score(model: Any, inputs: Any, answer_ids: tuple[int, int]) -> float:
    """Return z = logit("Yes") - logit("No") at the first answer position.

    Args:
        model: The loaded model (see ``load_model``).
        inputs: The model input of one clip (see ``clip_inputs``).
        answer_ids: The token IDs of "Yes" and "No" (see ``answer_token_ids``).

    Returns:
        The score z, computed in float32.
    """
    import torch

    yes_id, no_id = answer_ids
    with torch.inference_mode():
        output = model(**inputs.to(model.device), logits_to_keep=1)
    logits = output.logits[0, -1].float()
    return float(logits[yes_id] - logits[no_id])
