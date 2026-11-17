"""Sound event detection with PANNs (Cnn14_DecisionLevelMax, 527 AudioSet classes).

The event-extraction rules here are adapted from the audio-sort sidecar's
`sed.py` (~/ai-stack-wt-audiosort/audio-sort/sed.py) — copied rather than
imported, because the sidecar is a separate project. What is borrowed:

  * the label csv is `index,mid,display_name` and the row order IS the class
    order, so the label list is indexed by class id;
  * a frame's score above the threshold is ACTIVE, runs of active frames
    separated by a gap shorter than `gap_s` are ONE event, an event shorter
    than `min_len_s` is dropped, and an event's score is the peak inside it.

What is ours: the per-bin (0.5 s) maxima the moment scorer works on, and the
class list we care about (config/clipper.yaml).

The model runs on the CPU, in-process. The weights are expected
(~/panns_data) and are never downloaded: the check in `SoundEventDetector`
refuses to start if the checkpoint is missing, so the `panns_inference`
fallback (which would wget from Zenodo) can never fire.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

#: PANNs runs at 32 kHz with a 320-sample hop: 100 frames per second.
DEFAULT_FPS = 100
#: The smallest checkpoint `panns_inference` will accept (it checks this too).
MIN_CHECKPOINT_BYTES = 3e8


class SedError(Exception):
    """The sound event detector could not run."""


def load_labels(csv_path: str | Path) -> list[str]:
    """Read the PANNs class labels. The row order IS the class order."""
    path = Path(csv_path)
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise SedError(f"cannot read the label file {path}: {exc}") from exc
    labels: list[str] = []
    for row in rows[1:]:
        if len(row) < 3:
            continue
        labels.append(row[2])
    if not labels:
        raise SedError(f"no labels in {path.name}")
    return labels


def class_indices(labels: list[str], wanted: list[str]) -> dict[str, int]:
    """Map the class names we care about to their class ids."""
    lookup = {name: index for index, name in enumerate(labels)}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise SedError(
            f"the label file has no class named {', '.join(repr(m) for m in missing)}"
        )
    return {name: lookup[name] for name in wanted}


class SoundEventDetector:
    """PANNs on the CPU, in-process."""

    def __init__(self, weights_path: str | Path, labels_path: str | Path) -> None:
        self.weights_path = Path(weights_path)
        self.labels_path = Path(labels_path)
        if not self.weights_path.exists():
            raise SedError(
                f"the PANNs weights are missing: {self.weights_path} "
                "(put them there yourself; they are never downloaded)"
            )
        if self.weights_path.stat().st_size < MIN_CHECKPOINT_BYTES:
            raise SedError(
                f"the PANNs weights at {self.weights_path} look truncated "
                f"({self.weights_path.stat().st_size} bytes)"
            )
        self.labels = load_labels(self.labels_path)
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            try:
                import torch
                from panns_inference import SoundEventDetection
            except ImportError as exc:
                raise SedError(
                    f"the sound event detector needs torch and panns_inference: {exc}"
                ) from exc
            self._torch = torch
            self._detector = SoundEventDetection(
                device="cpu",
                checkpoint_path=str(self.weights_path),
            )
        return self._detector

    def framewise_scores(self, samples: np.ndarray) -> np.ndarray:
        """The framewise score matrix: (frames x 527)."""
        detector = self._get_detector()
        torch = self._torch
        clip = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            framewise = detector.inference(clip)
        matrix = np.asarray(framewise, dtype=np.float64)
        if matrix.ndim == 3:
            matrix = matrix[0]
        return matrix


def bin_max_scores(
    columns: dict[str, np.ndarray],
    n_bins: int,
    hop_seconds: float,
    fps: int = DEFAULT_FPS,
    min_score: float = 0.25,
) -> dict[str, np.ndarray]:
    """The maximum score of each class in each time bin.

    `columns` maps a class name to its framewise scores. A bin that never
    reaches `min_score` holds 0.0, so a class that is not really there
    contributes nothing to the score.
    """
    frames_per_bin = max(1, int(round(hop_seconds * fps)))
    out: dict[str, np.ndarray] = {}
    for name, column in columns.items():
        values = np.zeros(n_bins, dtype=np.float64)
        for index in range(n_bins):
            start = index * frames_per_bin
            if start >= column.shape[0]:
                break
            end = min(start + frames_per_bin, column.shape[0])
            peak = float(column[start:end].max())
            values[index] = peak if peak >= min_score else 0.0
        out[name] = values
    return out


def class_columns(
    framewise: np.ndarray,
    labels: list[str],
    wanted: list[str],
) -> dict[str, np.ndarray]:
    """Pull the columns of the classes we care about out of the framewise matrix.

    Only these columns are cached: the whole matrix is 500-odd classes wide
    and would be hundreds of megabytes per recording.
    """
    indices = class_indices(labels, wanted)
    out: dict[str, np.ndarray] = {}
    for name, class_id in indices.items():
        if class_id >= framewise.shape[1]:
            continue
        out[name] = framewise[:, class_id].astype(np.float32)
    return out


def extract_events(
    framewise: np.ndarray,
    labels: list[str],
    *,
    fps: int = DEFAULT_FPS,
    min_score: float = 0.25,
    gap_s: float = 0.15,
    min_len_s: float = 0.08,
    classes: list[str] | None = None,
) -> list[dict]:
    """Time-stamped sound events: `[{"start", "end", "label", "score"}, ...]`.

    The rules are the audio-sort ones (see the module docstring).
    """
    matrix = np.asarray(framewise, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise SedError(f"the framewise matrix must be 2-D, got shape {matrix.shape}")
    wanted = set(classes) if classes is not None else None
    events: list[dict] = []
    for class_id in range(matrix.shape[1]):
        name = labels[class_id] if class_id < len(labels) else str(class_id)
        if wanted is not None and name not in wanted:
            continue
        column = matrix[:, class_id]
        active = np.nonzero(column >= min_score)[0]
        if active.size == 0:
            continue
        merged: list[tuple[int, int]] = []
        for frame in active:
            frame = int(frame)
            if merged and (frame - merged[-1][1] - 1) / fps < gap_s:
                merged[-1] = (merged[-1][0], frame)
            else:
                merged.append((frame, frame))
        for first, last in merged:
            if (last - first + 1) / fps < min_len_s:
                continue
            events.append(
                {
                    "start": first / fps,
                    "end": (last + 1) / fps,
                    "label": name,
                    "score": float(column[first : last + 1].max()),
                }
            )
    events.sort(key=lambda event: (event["start"], event["end"], event["label"]))
    return events
