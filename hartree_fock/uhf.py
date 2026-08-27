from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .density import build_uhf_density
from .energy import uhf_electronic_energy
from .fock import build_uhf_fock
from .integrals import build_integrals
from .result import (
    ConvergenceInfo,
    EnergyResult,
    OrbitalResult,
    UHFResult,
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
    """Validate numerical settings used by the UHF-SCF loop."""
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


def _validate_uhf_system(
    molecule: Molecule,
    basis: BasisSet,
) -> tuple[int, int]:
    """Validate UHF occupations and return n_alpha and n_beta."""
    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta
    n_orbitals = len(basis)

    if n_alpha > n_orbitals:
        raise ValueError(
            "The basis set does not contain enough orbitals "
            "for all occupied alpha-spin orbitals."
        )

    if n_beta > n_orbitals:
        raise ValueError(
            "The basis set does not contain enough orbitals "
            "for all occupied beta-spin orbitals."
        )

    return n_alpha, n_beta


def _spin_density_rms_change(
    new_density_alpha: FloatArray,
    new_density_beta: FloatArray,
    old_density_alpha: FloatArray,
    old_density_beta: FloatArray,
) -> float:
    """Return one RMS change measured across both spin densities."""
    difference_alpha = (
        new_density_alpha
        - old_density_alpha
    )
    difference_beta = (
        new_density_beta
        - old_density_beta
    )

    squared_sum = (
        np.sum(difference_alpha * difference_alpha)
        + np.sum(difference_beta * difference_beta)
    )
    element_count = (
        difference_alpha.size
        + difference_beta.size
    )

    return float(
        np.sqrt(
            squared_sum / element_count
        )
    )


def _build_spin_occupations(
    n_orbitals: int,
    n_occupied: int,
) -> FloatArray:
    """Build a one-spin UHF orbital-occupation vector."""
    occupations = np.zeros(
        n_orbitals,
        dtype=np.float64,
    )
    occupations[:n_occupied] = 1.0

    return occupations


def run_uhf(
    molecule: Molecule,
    basis: BasisSet,
    *,
    max_iterations: int = 100,
    energy_tolerance: float = 1.0e-8,
    density_tolerance: float = 1.0e-6,
    overlap_eigenvalue_threshold: float = 1.0e-10,
) -> UHFResult:
    """
    Run an unrestricted Hartree-Fock calculation.

    UHF optimizes separate alpha- and beta-spin orbitals. Starting
    from a core-Hamiltonian guess, every SCF iteration performs

        P_alpha, P_beta
            -> F_alpha, F_beta
            -> C_alpha, C_beta
            -> P_alpha_new, P_beta_new.

    Both Fock matrices are rebuilt from the new spin densities before
    the energy is evaluated. Thus every recorded UHF energy uses
    density and Fock matrices belonging to the same SCF state.

    Parameters
    ----------
    molecule
        Molecular system with fixed nuclei. Its electron count and
        multiplicity determine ``n_alpha`` and ``n_beta``.
    basis
        Molecular AO basis set.
    max_iterations
        Maximum number of SCF updates after the core-Hamiltonian guess.
    energy_tolerance
        Convergence threshold for the absolute total-energy change.
    density_tolerance
        Convergence threshold for the RMS change across both alpha and
        beta AO density matrices.
    overlap_eigenvalue_threshold
        Smallest allowed overlap eigenvalue in the Roothaan solver.

    Returns
    -------
    UHFResult
        Final energy, separate alpha and beta orbital data, and
        convergence information. If the maximum iteration count is
        reached, the latest finite SCF state is returned with
        ``converged=False``.
    """
    _validate_scf_settings(
        max_iterations=max_iterations,
        energy_tolerance=energy_tolerance,
        density_tolerance=density_tolerance,
        overlap_eigenvalue_threshold=(
            overlap_eigenvalue_threshold
        ),
    )

    n_alpha, n_beta = _validate_uhf_system(
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

    # The same core-Hamiltonian orbitals initialize both spin channels.
    _, initial_coefficients = solve_roothaan(
        fock=core_hamiltonian,
        overlap=overlap,
        eigenvalue_threshold=(
            overlap_eigenvalue_threshold
        ),
    )

    (
        current_density_alpha,
        current_density_beta,
    ) = build_uhf_density(
        coefficients_alpha=initial_coefficients,
        coefficients_beta=initial_coefficients,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    (
        current_fock_alpha,
        current_fock_beta,
    ) = build_uhf_fock(
        core_hamiltonian=core_hamiltonian,
        density_alpha=current_density_alpha,
        density_beta=current_density_beta,
        electron_repulsion=electron_repulsion,
    )

    current_electronic_energy = (
        uhf_electronic_energy(
            density_alpha=current_density_alpha,
            density_beta=current_density_beta,
            core_hamiltonian=core_hamiltonian,
            fock_alpha=current_fock_alpha,
            fock_beta=current_fock_beta,
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
        _, coefficients_alpha = solve_roothaan(
            fock=current_fock_alpha,
            overlap=overlap,
            eigenvalue_threshold=(
                overlap_eigenvalue_threshold
            ),
        )
        _, coefficients_beta = solve_roothaan(
            fock=current_fock_beta,
            overlap=overlap,
            eigenvalue_threshold=(
                overlap_eigenvalue_threshold
            ),
        )

        (
            new_density_alpha,
            new_density_beta,
        ) = build_uhf_density(
            coefficients_alpha=coefficients_alpha,
            coefficients_beta=coefficients_beta,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )

        density_rms = _spin_density_rms_change(
            new_density_alpha=new_density_alpha,
            new_density_beta=new_density_beta,
            old_density_alpha=current_density_alpha,
            old_density_beta=current_density_beta,
        )

        # Rebuild both Fock matrices from the new spin densities before
        # evaluating the energy. This avoids mixing new densities with
        # Fock matrices from the preceding SCF state.
        (
            new_fock_alpha,
            new_fock_beta,
        ) = build_uhf_fock(
            core_hamiltonian=core_hamiltonian,
            density_alpha=new_density_alpha,
            density_beta=new_density_beta,
            electron_repulsion=electron_repulsion,
        )

        new_electronic_energy = (
            uhf_electronic_energy(
                density_alpha=new_density_alpha,
                density_beta=new_density_beta,
                core_hamiltonian=core_hamiltonian,
                fock_alpha=new_fock_alpha,
                fock_beta=new_fock_beta,
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

        current_density_alpha = (
            new_density_alpha
        )
        current_density_beta = (
            new_density_beta
        )
        current_fock_alpha = new_fock_alpha
        current_fock_beta = new_fock_beta
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

    # Canonicalize both final stored Fock matrices. At convergence, the
    # reconstructed densities differ from the stored densities by no
    # more than the requested SCF tolerance.
    (
        orbital_energies_alpha,
        coefficients_alpha,
    ) = solve_roothaan(
        fock=current_fock_alpha,
        overlap=overlap,
        eigenvalue_threshold=(
            overlap_eigenvalue_threshold
        ),
    )
    (
        orbital_energies_beta,
        coefficients_beta,
    ) = solve_roothaan(
        fock=current_fock_beta,
        overlap=overlap,
        eigenvalue_threshold=(
            overlap_eigenvalue_threshold
        ),
    )

    occupations_alpha = _build_spin_occupations(
        n_orbitals=orbital_energies_alpha.size,
        n_occupied=n_alpha,
    )
    occupations_beta = _build_spin_occupations(
        n_orbitals=orbital_energies_beta.size,
        n_occupied=n_beta,
    )

    return UHFResult(
        energy=EnergyResult(
            electronic=(
                current_electronic_energy
            ),
            nuclear_repulsion=nuclear_repulsion,
            total=current_total_energy,
        ),
        alpha=OrbitalResult(
            orbital_energies=(
                np.asarray(
                    orbital_energies_alpha,
                    dtype=np.float64,
                )
            ),
            coefficients=(
                np.asarray(
                    coefficients_alpha,
                    dtype=np.float64,
                )
            ),
            occupations=occupations_alpha,
            density=np.asarray(
                current_density_alpha,
                dtype=np.float64,
            ),
            fock=np.asarray(
                current_fock_alpha,
                dtype=np.float64,
            ),
        ),
        beta=OrbitalResult(
            orbital_energies=(
                np.asarray(
                    orbital_energies_beta,
                    dtype=np.float64,
                )
            ),
            coefficients=(
                np.asarray(
                    coefficients_beta,
                    dtype=np.float64,
                )
            ),
            occupations=occupations_beta,
            density=np.asarray(
                current_density_beta,
                dtype=np.float64,
            ),
            fock=np.asarray(
                current_fock_beta,
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
