from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """All the knobs of the environment. Lengths are in nucleotides (1 cell = 1 nt)."""

    h_max: int = 16               # canvas height: max number of helices
    l_max: int = 128              # canvas width: max helix length in nt

    l_staple_min: int = 20        # total staple length bounds
    l_staple_max: int = 60
    l_domain_min: int = 1         # min length of each of the two domains (1 = no constraint)
    crossover_period: int = 1     # crossover allowed only at columns x with x % period == 0 (1 = anywhere)

    min_rows: int = 2             # shape generator: min number of helices in a shape
    min_row_len: int = 8          # shape generator: min length of a helix
    shape_step: int = 6           # shape generator: max change of a row end between two rows

    max_steps_factor: float = 2.0  # truncation after factor * (n_cells // l_staple_min) steps

    def __post_init__(self) -> None:
        if self.h_max < 2:
            raise ValueError("h_max must be at least 2")
        if not 1 <= self.l_domain_min:
            raise ValueError("l_domain_min must be at least 1")
        if self.l_staple_min > self.l_staple_max:
            raise ValueError("l_staple_min must not exceed l_staple_max")
        if 2 * self.l_domain_min > self.l_staple_max:
            raise ValueError("two domains of length l_domain_min must fit in l_staple_max")
        if self.crossover_period < 1:
            raise ValueError("crossover_period must be at least 1")
        if not 1 <= self.min_row_len <= self.l_max:
            raise ValueError("min_row_len must be between 1 and l_max")
        if not 2 <= self.min_rows <= self.h_max:
            raise ValueError("min_rows must be between 2 and h_max")
        if self.shape_step < 0:
            raise ValueError("shape_step must be non-negative")

    @property
    def total_min(self) -> int:
        """Smallest total staple length that satisfies both length constraints."""
        return max(self.l_staple_min, 2 * self.l_domain_min)
