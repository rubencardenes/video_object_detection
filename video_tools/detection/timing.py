from collections import deque

import numpy as np


class InferenceTiming:
    """Bounded detector latency samples with warm-up and Tukey outlier filtering."""

    def __init__(self, backend: str = "ONNX") -> None:
        self.backend = backend
        self._warmup_remaining = 5
        self._samples: deque[float] = deque(maxlen=300)

    def add(self, milliseconds: float) -> None:
        if not np.isfinite(milliseconds) or milliseconds < 0:
            return
        if self._warmup_remaining:
            self._warmup_remaining -= 1
            return
        self._samples.append(milliseconds)

    def summary(self) -> tuple[float, int, int] | None:
        if len(self._samples) < 10:
            return None
        samples = np.asarray(self._samples)
        q1, q3 = np.percentile(samples, [25, 75])
        margin = 1.5 * (q3 - q1)
        kept = samples[(samples >= q1 - margin) & (samples <= q3 + margin)]
        return float(kept.mean()), len(kept), len(samples) - len(kept)

    def label(self) -> str:
        if self._warmup_remaining:
            return f"{self.backend}: warming up ({5 - self._warmup_remaining}/5)"
        summary = self.summary()
        if summary is None:
            return f"{self.backend}: sampling ({len(self._samples)}/10)"
        mean, count, excluded = summary
        return f"{self.backend}: {mean:.2f} ms · n={count} · excluded={excluded}"
