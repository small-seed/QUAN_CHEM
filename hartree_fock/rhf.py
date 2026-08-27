from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .density import build_rhf_density
from .energy import rhf_electronic_energy
from .fock import build_rhf_fock
from .integrals import build_integrals
from .result import (
    ConvergenceInfo,
    EnergyResult,
    OrbitalResult,
    RHFResult,
)
from .solver import solve_roothaan

if TYPE_CHECKING:
    from common.basis import BasisSet
    from common.molecule import Molecule


FloatArray = NDArray[np.float64]


def _validate_scf_settings(
    max_iterations: int,
    energy_tolerance: float,
    density_tolerance: float,
    overlap_eigenvalue_threshold: float,
) -> None:
    """Validate numerical settings used by the RHF-SCF loop."""
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, (int, np.integer))
    ):
        raise TypeError(
            "max_iterations must be an integer."
        )

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be at least one."
        )

    tolerances = (
        ("energy_tolerance", energy_tolerance),
        ("density_tolerance", density_tolerance),
        (
            "overlap_eigenvalue_threshold",
            overlap_eigenvalue_threshold,
        ),
    )

    for name, value in tolerances:
        if (
            not np.isscalar(value)
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"{name} must be a finite positive number."
            )


def _validate_rhf_system(
    molecule: Molecule,
    basis: BasisSet,
) -> int:
    """Validate a closed-shell RHF system and return its occupation."""
    if molecule.multiplicity != 1:
        raise ValueError(
            "RHF requires a closed-shell singlet "
            "with multiplicity equal to one."
        )

    if molecule.n_electrons % 2 != 0:
        raise ValueError(
            "RHF requires an even number of electrons."
        )

    n_occupied = molecule.n_electrons // 2

    if n_occupied > len(basis):
        raise ValueError(
            "The basis set does not contain enough orbitals "
            "for all occupied RHF orbitals."
        )

    return n_occupied


def _density_rms_change(
    new_density: FloatArray,
    old_density: FloatArray,
) -> float:
    """Return the root-mean-square change of an AO density matrix."""
    difference = new_density - old_density

    return float(
        np.sqrt(
            np.mean(difference * difference)
        )
    )


def _build_rhf_occupations(
    n_orbitals: int,
    n_occupied: int,
) -> FloatArray:
    """Build the RHF spatial-orbital occupation vector."""
    occupations = np.zeros(
        n_orbitals,
        dtype=np.float64,
    )
    occupations[:n_occupied] = 2.0

    return occupations


def run_rhf(
    molecule: Molecule,
    basis: BasisSet,
    *,
    max_iterations: int = 100,
    energy_tolerance: float = 1.0e-8,
    density_tolerance: float = 1.0e-6,
    overlap_eigenvalue_threshold: float = 1.0e-10,
) -> RHFResult:
    """
    Run a closed-shell restricted Hartree-Fock calculation.

    The initial density is obtained by solving the core-Hamiltonian
    generalized eigenvalue problem. Each SCF iteration then performs

        P -> F[P] -> C -> P_new.

    The Fock matrix is rebuilt from ``P_new`` before its energy is
    evaluated, so every recorded energy uses a density and Fock matrix
    belonging to the same SCF state.

    Parameters
    ----------
    molecule
        Molecular system with fixed nuclei. RHF requires a singlet
        system with an even number of electrons.
    basis
        Molecular AO basis set.
    max_iterations
        Maximum number of SCF updates after the core-Hamiltonian guess.
    energy_tolerance
        Convergence threshold for the absolute total-energy change.
    density_tolerance
        Convergence threshold for the RMS AO-density change.
    overlap_eigenvalue_threshold
        Smallest allowed overlap eigenvalue in the Roothaan solver.

    Returns
    -------
    RHFResult
        Final energy, orbital data, and convergence information. If the
        maximum iteration count is reached, the latest finite SCF state
        is returned with ``converged=False``.
    """
    _validate_scf_settings(
        max_iterations=max_iterations,
        energy_tolerance=energy_tolerance,
        density_tolerance=density_tolerance,
        overlap_eigenvalue_threshold=(
            overlap_eigenvalue_threshold
        ),
    )

    n_occupied = _validate_rhf_system(
        molecule=molecule,
        basis=basis,
    )

    (
        overlap,
        _,
        _,
        core_hamiltonian,
        electron_repulsion,
    ) = build_integrals(
        molecule=molecule,
        basis=basis,
    )

    nuclear_repulsion = (
        molecule.nuclear_repulsion_energy()
    )

    # Core-Hamiltonian guess:
    #
    #     H_core C^(0) = S C^(0) epsilon^(0).
    _, initial_coefficients = solve_roothaan(
        fock=core_hamiltonian,
        overlap=overlap,
        eigenvalue_threshold=(
            overlap_eigenvalue_threshold
        ),
    )

    current_density = build_rhf_density(
        coefficients=initial_coefficients,
        n_occupied=n_occupied,
    )

    current_fock = build_rhf_fock(
        core_hamiltonian=core_hamiltonian,
        density=current_density,
        electron_repulsion=electron_repulsion,
    )

    current_electronic_energy = (
        rhf_electronic_energy(
            density=current_density,
            core_hamiltonian=core_hamiltonian,
            fock=current_fock,
        )
    )
    current_total_energy = (
        current_electronic_energy
        + nuclear_repulsion
    )

    energy_history: list[float] = []
    density_rms_history: list[float] = []
    converged = False

    for _ in range(max_iterations):
        _, coefficients = solve_roothaan(
            fock=current_fock,
            overlap=overlap,
            eigenvalue_threshold=(
                overlap_eigenvalue_threshold
            ),
        )

        new_density = build_rhf_density(
            coefficients=coefficients,
            n_occupied=n_occupied,
        )

        density_rms = _density_rms_change(
            new_density=new_density,
            old_density=current_density,
        )

        # Rebuild F from the new density before evaluating the energy.
        # This avoids mixing P_new with F[P_old].
        new_fock = build_rhf_fock(
            core_hamiltonian=core_hamiltonian,
            density=new_density,
            electron_repulsion=electron_repulsion,
        )

        new_electronic_energy = (
            rhf_electronic_energy(
                density=new_density,
                core_hamiltonian=core_hamiltonian,
                fock=new_fock,
            )
        )
        new_total_energy = (
            new_electronic_energy
            + nuclear_repulsion
        )

        energy_change = abs(
            new_total_energy
            - current_total_energy
        )

        energy_history.append(
            new_total_energy
        )
        density_rms_history.append(
            density_rms
        )

        current_density = new_density
        current_fock = new_fock
        current_electronic_energy = (
            new_electronic_energy
        )
        current_total_energy = (
            new_total_energy
        )

        if (
            energy_change < energy_tolerance
            and density_rms < density_tolerance
        ):
            converged = True
            break

    # Canonicalize the final stored Fock matrix. At convergence, the
    # density reconstructed from these coefficients differs from the
    # stored density by no more than the requested SCF tolerance.
    orbital_energies, coefficients = (
        solve_roothaan(
            fock=current_fock,
            overlap=overlap,
            eigenvalue_threshold=(
                overlap_eigenvalue_threshold
            ),
        )
    )

    occupations = _build_rhf_occupations(
        n_orbitals=orbital_energies.size,
        n_occupied=n_occupied,
    )

    return RHFResult(
        energy=EnergyResult(
            electronic=(
                current_electronic_energy
            ),
            nuclear_repulsion=nuclear_repulsion,
            total=current_total_energy,
        ),
        orbitals=OrbitalResult(
            orbital_energies=(
                np.asarray(
                    orbital_energies,
                    dtype=np.float64,
                )
            ),
            coefficients=(
                np.asarray(
                    coefficients,
                    dtype=np.float64,
                )
            ),
            occupations=occupations,
            density=np.asarray(
                current_density,
                dtype=np.float64,
            ),
            fock=np.asarray(
                current_fock,
                dtype=np.float64,
            ),
        ),
        convergence=ConvergenceInfo(
            converged=converged,
            iterations=len(energy_history),
            energy_history=tuple(
                energy_history
            ),
            density_rms_history=tuple(
                density_rms_history
            ),
        ),
    )
