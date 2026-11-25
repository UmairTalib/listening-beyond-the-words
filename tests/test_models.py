from pathlib import Path

import pytest
import torch
from torch import nn

from listening_beyond_words.models import (
    ModelPin,
    WeightCheck,
    check_weights,
    load_pins,
    local_snapshot,
    snapshot_record,
)

PINS = Path(__file__).resolve().parents[1] / "configs" / "pilot" / "models.yaml"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def write_pins(folder: Path, text: str) -> Path:
    path = folder / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_pins_reads_repo_revision_and_ignore_patterns(tmp_path: Path) -> None:
    text = f"voxtral:\n  repo_id: org/model\n  revision: '{COMMIT}'\n  ignore: [big.bin]\n"
    pins = load_pins(write_pins(tmp_path, text))
    assert pins == {"voxtral": ModelPin("voxtral", "org/model", COMMIT, ("big.bin",))}


@pytest.mark.parametrize("revision", ["main", COMMIT[:12], COMMIT.upper()])
def test_load_pins_needs_a_full_commit_hash(tmp_path: Path, revision: str) -> None:
    path = write_pins(tmp_path, f"m:\n  repo_id: org/model\n  revision: '{revision}'\n")
    with pytest.raises(ValueError, match="40-character commit hash"):
        load_pins(path)


def test_load_pins_needs_repo_id_and_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a repo_id and a revision"):
        load_pins(write_pins(tmp_path, "m:\n  repo_id: org/model\n"))


def test_pilot_pins_voxtral_mini_at_a_full_commit() -> None:
    pin = load_pins(PINS)["voxtral"]
    assert pin.repo_id == "mistralai/Voxtral-Mini-3B-2507"
    assert len(pin.revision) == 40


def test_local_snapshot_finds_a_fetched_snapshot(tmp_path: Path) -> None:
    folder = tmp_path / "hub" / "models--org--model" / "snapshots" / COMMIT
    folder.mkdir(parents=True)
    assert local_snapshot(ModelPin("m", "org/model", COMMIT), tmp_path) == folder


def test_local_snapshot_says_how_to_fetch_a_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"scripts/fetch_models\.py"):
        local_snapshot(ModelPin("m", "org/model", COMMIT), tmp_path)


def test_snapshot_record_names_the_blob_of_each_linked_file(tmp_path: Path) -> None:
    repo = tmp_path / "hub" / "models--org--model"
    snapshot = repo / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (repo / "blobs").mkdir()
    for name, blob in (("model.safetensors", "b" * 64), ("config.json", "a" * 64)):
        (repo / "blobs" / blob).write_text(name, encoding="utf-8")
        (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / blob)
    (snapshot / "notes.txt").write_text("a file of its own, not a blob", encoding="utf-8")
    record = snapshot_record(ModelPin("m", "org/model", COMMIT), snapshot)
    assert record == {
        "repo_id": "org/model",
        "revision": COMMIT,
        "blobs": {"config.json": "a" * 64, "model.safetensors": "b" * 64},
    }
    assert list(record["blobs"]) == ["config.json", "model.safetensors"]  # by name


def test_pilot_pins_kokoro_and_faster_whisper() -> None:
    pins = load_pins(PINS)
    assert pins["kokoro"].repo_id == "hexgrad/Kokoro-82M"
    assert pins["kokoro"].ignore == ("samples/*", "eval/*")
    assert pins["faster_whisper"].repo_id == "Systran/faster-whisper-large-v3"


class Tiny(nn.Module):
    """A linear layer and a computed (non-persistent) buffer, like Albert's position indices."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("positions", torch.arange(4), persistent=False)


def weights_file(model: nn.Module) -> dict[str, torch.Tensor]:
    """A weights file for the model whose values all differ from its initial ones."""
    return {name: tensor + 1 for name, tensor in model.state_dict().items()}


def test_check_weights_passes_when_the_loader_took_every_tensor() -> None:
    model = Tiny()
    state = weights_file(model)
    model.load_state_dict(state)
    check = check_weights(model, state)
    assert check == WeightCheck(
        tensors=2, missing=(), unused=(), changed=(), fixed_buffers=(), identity_norms=()
    )
    assert check.ok


def test_check_weights_finds_weights_that_the_loader_skipped() -> None:
    model = Tiny()
    state = weights_file(model)  # the loader never copied it in
    check = check_weights(model, state)
    assert check.changed == ("linear.bias", "linear.weight")
    assert not check.ok


def test_check_weights_finds_a_renamed_tensor_left_out_by_a_lenient_loader() -> None:
    model = Tiny()
    state = weights_file(model)
    state["linear.weights"] = state.pop("linear.weight")
    model.load_state_dict(state, strict=False)  # loads the bias, skips the weight silently
    check = check_weights(model, state)
    assert check.missing == ("linear.weight",)
    assert check.unused == ("linear.weights",)
    assert check.changed == ()
    assert not check.ok


def test_check_weights_accepts_a_computed_buffer_with_the_same_values() -> None:
    model = Tiny()
    state = {**weights_file(model), "positions": torch.arange(4)}
    model.load_state_dict(state, strict=False)
    check = check_weights(model, state)
    assert check.fixed_buffers == ("positions",)
    assert check.unused == ()
    assert check.tensors == 3  # counted in the file, not in the model
    assert check.ok


def test_check_weights_refuses_a_computed_buffer_with_other_values() -> None:
    model = Tiny()
    state = {**weights_file(model), "positions": torch.arange(4) + 1}
    model.load_state_dict(state, strict=False)
    check = check_weights(model, state)
    assert check.unused == ("positions",)
    assert check.fixed_buffers == ()
    assert not check.ok


@pytest.mark.filterwarnings("ignore:.*weight_norm.*:FutureWarning")
@pytest.mark.parametrize(
    "weight_norm", [nn.utils.weight_norm, nn.utils.parametrizations.weight_norm]
)
def test_check_weights_follows_the_names_of_both_weight_norm_styles(weight_norm) -> None:
    # Kokoro 0.9.4 uses the old style; its file names the weights weight_g and weight_v.
    state = weights_file(nn.utils.weight_norm(nn.Conv1d(2, 2, 3)))
    model = weight_norm(nn.Conv1d(2, 2, 3))
    model.load_state_dict(state)
    assert check_weights(model, state).ok


class TwoParts(nn.Module):
    """Two parts that a loader fills one by one from the file, as Kokoro's does."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 2)
        self.decoder = nn.Linear(3, 2)


def test_check_weights_finds_a_part_that_the_file_lacks() -> None:
    model = TwoParts()
    part = weights_file(model.encoder)
    model.encoder.load_state_dict(part)  # the loader loops over the file's parts only
    check = check_weights(model, {f"encoder.{name}": tensor for name, tensor in part.items()})
    assert (check.missing, check.unused, check.changed) == (
        ("decoder.bias", "decoder.weight"),
        (),
        (),
    )
    assert not check.ok


def test_check_weights_finds_buffers_that_the_loader_skipped() -> None:
    model = nn.BatchNorm1d(3)
    state = weights_file(model)
    model.load_state_dict({name: state[name] for name in ("weight", "bias")}, strict=False)
    check = check_weights(model, state)
    assert check.changed == ("num_batches_tracked", "running_mean", "running_var")
    assert not check.ok


class Styled(nn.Module):
    """A linear layer and an instance norm with scale and shift, like Kokoro's AdaIN norms."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)
        self.norm = nn.InstanceNorm1d(2, affine=True)


def styled_without_norm_weights() -> tuple[Styled, dict[str, torch.Tensor]]:
    """A Styled model loaded from a file that names only the linear layer."""
    model = Styled()
    state = {f"fc.{name}": tensor for name, tensor in weights_file(model.fc).items()}
    model.load_state_dict(state, strict=False)
    return model, state


def test_check_weights_accepts_instance_norm_weights_left_at_identity() -> None:
    model, state = styled_without_norm_weights()
    check = check_weights(model, state)
    assert check.identity_norms == ("norm.bias", "norm.weight")
    assert check.missing == ()
    assert check.ok


@pytest.mark.parametrize(("name", "value"), [("norm.weight", 2.0), ("norm.bias", 0.5)])
def test_check_weights_refuses_instance_norm_weights_away_from_identity(
    name: str, value: float
) -> None:
    model, state = styled_without_norm_weights()
    with torch.no_grad():
        model.get_parameter(name).fill_(value)
    check = check_weights(model, state)
    assert check.missing == (name,)
    assert not check.ok


def test_check_weights_still_fails_on_other_missing_weights_beside_identity_norms() -> None:
    model, state = styled_without_norm_weights()
    del state["fc.bias"]
    check = check_weights(model, state)
    assert check.identity_norms == ("norm.bias", "norm.weight")
    assert check.missing == ("fc.bias",)
    assert not check.ok
