from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Accuracy:
    correct: int
    total: int

    @property
    def acc(self) -> float:
        if self.total <= 0:
            return 0.0
        return float(self.correct) / float(self.total)

