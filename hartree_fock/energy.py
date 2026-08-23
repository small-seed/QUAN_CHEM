from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from common.molecule import Molecule


FloatArray = NDArray[np.float64]


def _validate_square_matrix(
    matrix: FloatArray,
    name: str,
) -> FloatArray:
    """
    Validate and return a finite square matrix.
    """
    array = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(
            f"The {name} matrix must be square."
        )

    if array.shape[0] == 0:
        raise ValueError(
            f"The {name} matrix cannot be empty."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"The {name} matrix must contain only finite values."
        )

    return array


def _validate_energy(
    energy: float,
    name: str,
) -> float:
    """
    Validate and return a finite scalar energy.
    """
    try:
        value = float(energy)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"The {name} energy must be a real scalar."
        ) from error

    if not np.isfinite(value):
        raise ValueError(
            f"The {name} energy must be finite."
        )

    return value


def rhf_electronic_energy(
    density: FloatArray,
    core_hamiltonian: FloatArray,
    fock: FloatArray,
) -> float:
    """
    Compute the restricted Hartree-Fock electronic energy.

    This function assumes the spin-summed RHF density convention used
    by ``density.build_rhf_density``:

        P[mu, nu]
            = 2 sum_i^occupied
              C[mu, i] C[nu, i].

    With this convention, the electronic energy is

        E_elec
            = 1/2 sum_mu,nu
              P[mu, nu]
              (H_core[nu, mu] + F[nu, mu]).

    The factor of one half prevents the electron-electron contribution,
    which is already contained in the Fock matrix, from being counted
    twice.

    Parameters
    ----------
    density
        Spin-summed RHF AO density matrix P.
    core_hamiltonian
        AO core Hamiltonian H_core = T + V.
    fock
        Converged or current-iteration RHF AO Fock matrix F.

    Returns
    -------
    float
        RHF electronic energy in atomic units.
    """
    density_matrix = _validate_square_matrix(
        matrix=density,
        name="density",
    )

    core_matrix = _validate_square_matrix(
        matrix=core_hamiltonian,
        name="core Hamiltonian",
    )

    fock_matrix = _validate_square_matrix(
        matrix=fock,
        name="fock",
    )

    if (
        density_matrix.shape != core_matrix.shape
        or density_matrix.shape != fock_matrix.shape
    ):
        raise ValueError(
            "Density, core Hamiltonian, and Fock matrices "
            "must have equal shapes."
        )

    energy = 0.5 * np.einsum(
        "mn,nm->",
        density_matrix,
        core_matrix + fock_matrix,
        optimize=True,
    )

    return float(energy)


def uhf_electronic_energy(
    density_alpha: FloatArray,
    density_beta: FloatArray,
    core_hamiltonian: FloatArray,
    fock_alpha: FloatArray,
    fock_beta: FloatArray,
) -> float:
    """
    Compute the unrestricted Hartree-Fock electronic energy.

    This function assumes the one-spin density convention used by
    ``density.build_uhf_density``:

        P_alpha[mu, nu]
            = sum_i^n_alpha
              C_alpha[mu, i] C_alpha[nu, i]

        P_beta[mu, nu]
            = sum_i^n_beta
              C_beta[mu, i] C_beta[nu, i].

    Neither density contains a factor of two. The UHF electronic
    energy is therefore

        E_elec
            = 1/2 sum_mu,nu
              P_alpha[mu, nu]
              (H_core[nu, mu] + F_alpha[nu, mu])

              + 1/2 sum_mu,nu
              P_beta[mu, nu]
              (H_core[nu, mu] + F_beta[nu, mu]).

    Parameters
    ----------
    density_alpha
        Alpha-spin AO density matrix P_alpha.
    density_beta
        Beta-spin AO density matrix P_beta.
    core_hamiltonian
        AO core Hamiltonian H_core = T + V.
    fock_alpha
        Alpha-spin AO Fock matrix F_alpha.
    fock_beta
        Beta-spin AO Fock matrix F_beta.

    Returns
    -------
    float
        UHF electronic energy in atomic units.
    """
    alpha_density = _validate_square_matrix(
        matrix=density_alpha,
        name="alpha density",
    )

    beta_density = _validate_square_matrix(
        matrix=density_beta,
        name="beta density",
    )

    core_matrix = _validate_square_matrix(
        matrix=core_hamiltonian,
        name="core Hamiltonian",
    )

    alpha_fock = _validate_square_matrix(
        matrix=fock_alpha,
        name="alpha fock",
    )

    beta_fock = _validate_square_matrix(
        matrix=fock_beta,
        name="beta fock",
    )

    matrix_shape = core_matrix.shape

    if any(
        matrix.shape != matrix_shape
        for matrix in (
            alpha_density,
            beta_density,
            alpha_fock,
            beta_fock,
        )
    ):
        raise ValueError(
            "Alpha density, beta density, core Hamiltonian, "
            "alpha Fock, and beta Fock matrices must have "
            "equal shapes."
        )

    alpha_energy = np.einsum(
        "mn,nm->",
        alpha_density,
        core_matrix + alpha_fock,
        optimize=True,
    )

    beta_energy = np.einsum(
        "mn,nm->",
        beta_density,
        core_matrix + beta_fock,
        optimize=True,
    )

    return float(
        0.5 * (alpha_energy + beta_energy)
    )


def nuclear_repulsion_energy(
    molecule: Molecule,
) -> float:
    """
    Return the classical nuclear-repulsion energy of a molecule.

    ``Molecule`` owns the nuclear coordinates and charges, so the
    actual pairwise calculation remains in
    ``Molecule.nuclear_repulsion_energy``:

        E_NN = sum_A<B Z_A Z_B / R_AB.
    """
    method = getattr(
        molecule,
        "nuclear_repulsion_energy",
        None,
    )

    if method is None or not callable(method):
        raise TypeError(
            "molecule must provide a callable "
            "nuclear_repulsion_energy method."
        )

    energy = _validate_energy(
        energy=method(),
        name="nuclear-repulsion",
    )

    if energy < 0.0:
        raise ValueError(
            "The nuclear-repulsion energy cannot be negative."
        )

    return energy


def total_energy(
    electronic_energy: float,
    nuclear_repulsion: float,
) -> float:
    """
    Compute the Born-Oppenheimer Hartree-Fock total energy.

        E_total = E_elec + E_NN

    Both input energies and the returned value are in atomic units.
    """
    electronic = _validate_energy(
        energy=electronic_energy,
        name="electronic",
    )

    nuclear = _validate_energy(
        energy=nuclear_repulsion,
        name="nuclear-repulsion",
    )

    if nuclear < 0.0:
        raise ValueError(
            "The nuclear-repulsion energy cannot be negative."
        )

    return float(electronic + nuclear)