from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnergyResult:
    """Store the energy components of a completed SCF calculation.

    All values are supplied by ``energy.py`` and are expressed in
    atomic units. This class stores the results only; it does not
    recompute or validate the total energy.

    Attributes
    ----------
    electronic
        Hartree-Fock electronic energy.
    nuclear_repulsion
        Classical nuclear-repulsion energy.
    total
        Born-Oppenheimer total energy, previously computed as
        ``electronic + nuclear_repulsion`` by ``energy.total_energy``.
    """

    electronic: float
    nuclear_repulsion: float
    total: float


@dataclass(frozen=True, slots=True)
class ConvergenceInfo:
    """Store information about the SCF convergence process.

    Attributes
    ----------
    converged
        Whether the SCF convergence criteria were satisfied.
    iterations
        Number of SCF iterations performed.
    energy_history
        Total or electronic energy recorded after each iteration,
        according to the convention used by the SCF driver.
    density_rms_history
        RMS density-matrix change recorded after each iteration.
    """

    converged: bool
    iterations: int
    energy_history: tuple[float, ...] = ()
    density_rms_history: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class OrbitalResult:
    """Store the final orbital data for one spin channel.

    RHF uses one ``OrbitalResult`` shared by both spin channels. UHF
    uses separate alpha and beta instances.

    Attributes
    ----------
    orbital_energies
        Final orbital energies with shape ``(K,)``.
    coefficients
        Final AO-to-MO coefficient matrix with shape ``(K, K)``.
    occupations
        Orbital occupations with shape ``(K,)``. RHF occupations are
        normally zero or two, while one-spin UHF occupations are
        normally zero or one.
    density
        Final AO density matrix with shape ``(K, K)``.
    fock
        Final AO Fock matrix with shape ``(K, K)``.
    """

    orbital_energies: FloatArray
    coefficients: FloatArray
    occupations: FloatArray
    density: FloatArray
    fock: FloatArray


@dataclass(frozen=True, slots=True)
class RHFResult:
    """Complete result of a restricted Hartree-Fock calculation."""

    energy: EnergyResult
    orbitals: OrbitalResult
    convergence: ConvergenceInfo


@dataclass(frozen=True, slots=True)
class UHFResult:
    """Complete result of an unrestricted Hartree-Fock calculation."""

    energy: EnergyResult
    alpha: OrbitalResult
    beta: OrbitalResult
    convergence: ConvergenceInfo