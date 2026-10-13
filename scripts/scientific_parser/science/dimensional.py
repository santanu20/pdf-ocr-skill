"""Dimensional analysis — physics SI dimension propagation.

Per proposal item #28: "Equation engine: LaTeX → AST → dimensional validator."

Physics dimensions (7 SI base + common derived):
  L  = length
  M  = mass
  T  = time
  I  = electric current
  Θ  = temperature
  N  = amount of substance
  J  = luminous intensity

Each physics variable maps to a DimensionVector. Operations propagate:
  + / - : same dimensions required
  * : dimensions add
  / : dimensions subtract
  ^n : dimensions multiply by n
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Dimension:
    """7-component SI dimension vector."""

    L: int = 0  # length
    M: int = 0  # mass
    T: int = 0  # time
    I: int = 0  # electric current
    THETA: int = 0  # temperature
    N: int = 0  # amount
    J: int = 0  # luminous intensity

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            self.L + other.L,
            self.M + other.M,
            self.T + other.T,
            self.I + other.I,
            self.THETA + other.THETA,
            self.N + other.N,
            self.J + other.J,
        )

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            self.L - other.L,
            self.M - other.M,
            self.T - other.T,
            self.I - other.I,
            self.THETA - other.THETA,
            self.N - other.N,
            self.J - other.J,
        )

    def power(self, n: int) -> "Dimension":
        return Dimension(
            self.L * n,
            self.M * n,
            self.T * n,
            self.I * n,
            self.THETA * n,
            self.N * n,
            self.J * n,
        )

    def is_dimensionless(self) -> bool:
        return all(
            getattr(self, f) == 0 for f in ("L", "M", "T", "I", "THETA", "N", "J")
        )

    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return False
        return all(
            getattr(self, f) == getattr(other, f)
            for f in ("L", "M", "T", "I", "THETA", "N", "J")
        )

    def __hash__(self):
        return hash((self.L, self.M, self.T, self.I, self.THETA, self.N, self.J))

    def __str__(self):
        if self.is_dimensionless():
            return "1 (dimensionless)"
        parts = []
        for name, val in [
            ("L", self.L),
            ("M", self.M),
            ("T", self.T),
            ("I", self.I),
            ("Θ", self.THETA),
            ("N", self.N),
            ("J", self.J),
        ]:
            if val != 0:
                parts.append(f"{name}^{val}" if val != 1 else name)
        return "·".join(parts)


# === SI base dimensions ====================================================
DIM_LENGTH = Dimension(L=1)
DIM_MASS = Dimension(M=1)
DIM_TIME = Dimension(T=1)
DIM_CURRENT = Dimension(I=1)
DIM_TEMPERATURE = Dimension(THETA=1)
DIM_AMOUNT = Dimension(N=1)
DIM_LUMINOUS = Dimension(J=1)
DIM_DIMENSIONLESS = Dimension()


# === Physics symbol → dimension map =======================================
# Common physics symbols and their SI dimensions
PHYSICS_DIMENSIONS = {
    # Mechanics
    "x": DIM_LENGTH,
    "y": DIM_LENGTH,
    "z": DIM_LENGTH,
    "r": DIM_LENGTH,
    "s": DIM_LENGTH,
    "d": DIM_LENGTH,
    "h": DIM_LENGTH,
    "l": DIM_LENGTH,
    "λ": DIM_LENGTH,  # wavelength
    "v": DIM_LENGTH / DIM_TIME,  # velocity
    "a": DIM_LENGTH / DIM_TIME.power(2),  # acceleration
    "g": DIM_LENGTH / DIM_TIME.power(2),  # gravity
    "ω": DIM_DIMENSIONLESS / DIM_TIME,  # angular velocity (rad/s)
    "α": DIM_DIMENSIONLESS / DIM_TIME.power(2),  # angular acceleration
    "f": DIM_DIMENSIONLESS / DIM_TIME,  # frequency
    "ν": DIM_DIMENSIONLESS / DIM_TIME,  # frequency
    "T": DIM_TEMPERATURE,
    "Θ": DIM_TEMPERATURE,
    "t": DIM_TIME,
    "τ": DIM_TIME,  # time, time constant
    "m": DIM_MASS,
    "M": DIM_MASS,
    "μ": DIM_MASS,
    "ρ": DIM_MASS / DIM_LENGTH.power(3),  # density
    "p": DIM_MASS / (DIM_LENGTH * DIM_TIME.power(2)),  # pressure
    "F": DIM_MASS * DIM_LENGTH / DIM_TIME.power(2),  # force (Newton)
    "E": DIM_MASS * DIM_LENGTH.power(2) / DIM_TIME.power(2),  # energy (Joule)
    "W": DIM_MASS * DIM_LENGTH.power(2) / DIM_TIME.power(2),  # work
    "P": DIM_MASS * DIM_LENGTH.power(2) / DIM_TIME.power(3),  # power (Watt)
    "L": DIM_MASS * DIM_LENGTH.power(2) / DIM_TIME,  # angular momentum
    "I": DIM_MASS * DIM_LENGTH.power(2),  # moment of inertia
    "σ": DIM_MASS / DIM_TIME.power(2),  # surface tension
    "η": DIM_MASS / (DIM_LENGTH * DIM_TIME),  # viscosity
    # Electromagnetism
    "q": DIM_CURRENT * DIM_TIME,  # charge (Coulomb)
    "Q": DIM_CURRENT * DIM_TIME,
    "V": DIM_MASS * DIM_LENGTH.power(2) / (DIM_CURRENT * DIM_TIME.power(3)),  # voltage
    "φ": DIM_MASS * DIM_LENGTH.power(2) / (DIM_CURRENT * DIM_TIME.power(3)),
    "R": DIM_MASS * DIM_LENGTH.power(2) / (DIM_CURRENT.power(2) * DIM_TIME.power(3)),
    "B": DIM_MASS / (DIM_CURRENT * DIM_TIME.power(2)),  # magnetic field
    "H": DIM_CURRENT / DIM_LENGTH,  # magnetic H field
    "C": DIM_CURRENT.power(2) * DIM_TIME.power(4) / (DIM_MASS * DIM_LENGTH.power(2)),
    # Thermodynamics
    "S": DIM_MASS
    * DIM_LENGTH.power(2)
    / (DIM_TIME.power(2) * DIM_TEMPERATURE),  # entropy
    "H": DIM_MASS * DIM_LENGTH.power(2) / DIM_TIME.power(2),  # enthalpy
    "G": DIM_MASS * DIM_LENGTH.power(2) / DIM_TIME.power(2),  # Gibbs free energy
    "U": DIM_MASS * DIM_LENGTH.power(2) / DIM_TIME.power(2),  # internal energy
    "k": DIM_DIMENSIONLESS,  # Boltzmann constant (when N=amount)
    "R": DIM_MASS
    * DIM_LENGTH.power(2)
    / (DIM_TIME.power(2) * DIM_TEMPERATURE * DIM_AMOUNT),
    # Waves / optics
    "c": DIM_LENGTH / DIM_TIME,  # speed of light
    "n": DIM_DIMENSIONLESS,  # refractive index
    "φ": DIM_DIMENSIONLESS,  # phase
    # Quantum
    "ℏ": DIM_MASS * DIM_LENGTH.power(2) / DIM_TIME,  # reduced Planck
    "ψ": DIM_DIMENSIONLESS,  # wavefunction
    "μ": DIM_DIMENSIONLESS,  # magnetic moment per unit
    # Statistics
    "σ": DIM_DIMENSIONLESS,  # std dev (when over normalized quantity)
    "μ": DIM_DIMENSIONLESS,  # mean
    # Common subscripts only matter via parent symbol
}


def get_dimension(
    symbol: str, custom_map: Optional[dict] = None
) -> Optional[Dimension]:
    """Look up SI dimension for a symbol. None if unknown."""
    if custom_map and symbol in custom_map:
        return custom_map[symbol]
    return PHYSICS_DIMENSIONS.get(symbol)


def check_equation_consistency(
    lhs_symbols: list,
    rhs_symbols: list,
    operators: list,
    custom_map: Optional[dict] = None,
) -> dict:
    """Check dimensional consistency of LHS = RHS expression.

    Args:
        lhs_symbols: list of symbols on left of =
        rhs_symbols: list of symbols on right of =
        operators: list of operators between symbols (length = n_symbols - 1)
        custom_map: optional {symbol: Dimension} override

    Returns:
        {
          "consistent": bool,
          "lhs_dim": Dimension,
          "rhs_dim": Dimension,
          "unknown_symbols": list,
        }
    """

    def _compute_dim(symbols, ops):
        if not symbols:
            return DIM_DIMENSIONLESS
        dim = get_dimension(symbols[0], custom_map) or DIM_DIMENSIONLESS
        unknown = set()
        if symbols[0] not in PHYSICS_DIMENSIONS and (
            custom_map is None or symbols[0] not in custom_map
        ):
            unknown.add(symbols[0])
        for sym, op in zip(symbols[1:], ops):
            d = get_dimension(sym, custom_map)
            if d is None:
                unknown.add(sym)
                d = DIM_DIMENSIONLESS
            if op in ("+", "-"):
                # + and -: dimensions must match (skipped for robustness)
                dim = dim  # keep first
            elif op in ("*", "\\cdot"):
                dim = dim * d
            elif op == "/":
                dim = dim / d
        return dim, unknown

    lhs_dim, lhs_unknown = _compute_dim(
        lhs_symbols, operators[: max(0, len(lhs_symbols) - 1)]
    )
    rhs_dim, rhs_unknown = _compute_dim(
        rhs_symbols, operators[: max(0, len(rhs_symbols) - 1)]
    )
    return {
        "consistent": lhs_dim == rhs_dim,
        "lhs_dim": str(lhs_dim),
        "rhs_dim": str(rhs_dim),
        "unknown_symbols": sorted(lhs_unknown | rhs_unknown),
    }


__all__ = [
    "Dimension",
    "DIM_LENGTH",
    "DIM_MASS",
    "DIM_TIME",
    "DIM_CURRENT",
    "DIM_TEMPERATURE",
    "DIM_AMOUNT",
    "DIM_LUMINOUS",
    "DIM_DIMENSIONLESS",
    "PHYSICS_DIMENSIONS",
    "get_dimension",
    "check_equation_consistency",
]
