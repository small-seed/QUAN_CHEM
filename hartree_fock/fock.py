from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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


def _validate_eri_tensor(
    electron_repulsion: FloatArray,
) -> FloatArray:
    """
    Validate and return an AO electron-repulsion tensor.

    The tensor convention is chemists' notation:

        eri[mu, nu, lam, sig] = (mu nu | lam sig).
    """
    eri = np.asarray(
        electron_repulsion,
        dtype=np.float64,
    )

    if eri.ndim != 4:
        raise ValueError(
            "The electron-repulsion tensor must be four-dimensional."
        )

    if any(size == 0 for size in eri.shape):
        raise ValueError(
            "The electron-repulsion tensor cannot be empty."
        )

    n_basis = eri.shape[0]

    if eri.shape != (
        n_basis,
        n_basis,
        n_basis,
        n_basis,
    ):
        raise ValueError(
            "All four electron-repulsion tensor dimensions "
            "must have equal size."
        )

    if not np.all(np.isfinite(eri)):
        raise ValueError(
            "The electron-repulsion tensor must contain "
            "only finite values."
        )

    return eri


def _validate_fock_inputs(
    core_hamiltonian: FloatArray,
    density: FloatArray,
    electron_repulsion: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Validate matrices and ERI tensor used to construct a Fock matrix.
    """
    core_matrix = _validate_square_matrix(
        matrix=core_hamiltonian,
        name="core Hamiltonian",
    )

    density_matrix = _validate_square_matrix(
        matrix=density,
        name="density",
    )

    eri = _validate_eri_tensor(
        electron_repulsion
    )

    if core_matrix.shape != density_matrix.shape:
        raise ValueError(
            "Core Hamiltonian and density matrices "
            "must have equal shapes."
        )

    n_basis = core_matrix.shape[0]

    if eri.shape != (
        n_basis,
        n_basis,
        n_basis,
        n_basis,
    ):
        raise ValueError(
            "Electron-repulsion tensor dimensions must match "
            "the AO matrix dimension."
        )

    return (
        core_matrix,
        density_matrix,
        eri,
    )


def build_coulomb_matrix(
    density: FloatArray,
    electron_repulsion: FloatArray,
) -> FloatArray:
    """
    Build the AO Coulomb matrix.

    The ERI convention is:

        eri[mu, nu, lam, sig] = (mu nu | lam sig)

    and therefore:

        J[mu, nu]
            = sum_lam,sig
              P[lam, sig]
              (mu nu | lam sig).

    The meaning of P depends on the caller:

    - RHF:
        P is the spin-summed density containing the factor of two.

    - UHF:
        P is normally the total density:

            P_total = P_alpha + P_beta.
    """
    density_matrix = _validate_square_matrix(
        matrix=density,
        name="density",
    )

    eri = _validate_eri_tensor(
        electron_repulsion
    )

    n_basis = density_matrix.shape[0]

    if eri.shape != (
        n_basis,
        n_basis,
        n_basis,
        n_basis,
    ):
        raise ValueError(
            "Electron-repulsion tensor dimensions must match "
            "the density-matrix dimension."
        )

    coulomb = np.einsum(
        "ls,mnls->mn",
        density_matrix,
        eri,
        optimize=True,
    )

    return np.asarray(
        coulomb,
        dtype=np.float64,
    )


def build_exchange_matrix(
    density: FloatArray,
    electron_repulsion: FloatArray,
) -> FloatArray:
    """
    Build the AO exchange matrix.

    With chemists' notation:

        eri[mu, nu, lam, sig] = (mu nu | lam sig),

    the exchange matrix is:

        K[mu, nu]
            = sum_lam,sig
              P[lam, sig]
              (mu lam | nu sig).

    Hence the ERI indices used here are:

        eri[mu, lam, nu, sig].
    """
    density_matrix = _validate_square_matrix(
        matrix=density,
        name="density",
    )

    eri = _validate_eri_tensor(
        electron_repulsion
    )

    n_basis = density_matrix.shape[0]

    if eri.shape != (
        n_basis,
        n_basis,
        n_basis,
        n_basis,
    ):
        raise ValueError(
            "Electron-repulsion tensor dimensions must match "
            "the density-matrix dimension."
        )

    exchange = np.einsum(
        "ls,mlns->mn",
        density_matrix,
        eri,
        optimize=True,
    )

    return np.asarray(
        exchange,
        dtype=np.float64,
    )


def build_rhf_fock(
    core_hamiltonian: FloatArray,
    density: FloatArray,
    electron_repulsion: FloatArray,
) -> FloatArray:
    """
    Build the restricted Hartree-Fock AO Fock matrix.

    This function assumes the RHF density convention used by
    density.build_rhf_density:

        P[mu, nu]
            = 2 sum_i^occupied
              C[mu, i] C[nu, i].

    Therefore:

        J[mu, nu]
            = sum_lam,sig
              P[lam, sig]
              (mu nu | lam sig)

        K[mu, nu]
            = sum_lam,sig
              P[lam, sig]
              (mu lam | nu sig)

    and the RHF Fock matrix is:

        F = H_core + J - 1/2 K.
    """
    (
        core_matrix,
        density_matrix,
        eri,
    ) = _validate_fock_inputs(
        core_hamiltonian=core_hamiltonian,
        density=density,
        electron_repulsion=electron_repulsion,
    )

    coulomb = build_coulomb_matrix(
        density=density_matrix,
        electron_repulsion=eri,
    )

    exchange = build_exchange_matrix(
        density=density_matrix,
        electron_repulsion=eri,
    )

    fock = (
        core_matrix
        + coulomb
        - 0.5 * exchange
    )

    return np.asarray(
        fock,
        dtype=np.float64,
    )


def build_uhf_fock(
    core_hamiltonian: FloatArray,
    density_alpha: FloatArray,
    density_beta: FloatArray,
    electron_repulsion: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Build alpha- and beta-spin unrestricted Hartree-Fock Fock matrices.

    The one-spin UHF densities contain no factor of two:

        P_alpha[mu, nu]
            = sum_i^n_alpha
              C_alpha[mu, i] C_alpha[nu, i]

        P_beta[mu, nu]
            = sum_i^n_beta
              C_beta[mu, i] C_beta[nu, i].

    The total Coulomb density is:

        P_total = P_alpha + P_beta.

    Coulomb interaction depends on all electrons, while exchange
    occurs only between electrons with the same spin:

        F_alpha
            = H_core
              + J[P_total]
              - K[P_alpha]

        F_beta
            = H_core
              + J[P_total]
              - K[P_beta].
    """
    core_matrix = _validate_square_matrix(
        matrix=core_hamiltonian,
        name="core Hamiltonian",
    )

    alpha_matrix = _validate_square_matrix(
        matrix=density_alpha,
        name="alpha density",
    )

    beta_matrix = _validate_square_matrix(
        matrix=density_beta,
        name="beta density",
    )

    eri = _validate_eri_tensor(
        electron_repulsion
    )

    if (
        core_matrix.shape != alpha_matrix.shape
        or core_matrix.shape != beta_matrix.shape
    ):
        raise ValueError(
            "Core Hamiltonian, alpha density, and beta density "
            "matrices must have equal shapes."
        )

    n_basis = core_matrix.shape[0]

    if eri.shape != (
        n_basis,
        n_basis,
        n_basis,
        n_basis,
    ):
        raise ValueError(
            "Electron-repulsion tensor dimensions must match "
            "the AO matrix dimension."
        )

    total_density = (
        alpha_matrix
        + beta_matrix
    )

    coulomb = build_coulomb_matrix(
        density=total_density,
        electron_repulsion=eri,
    )

    exchange_alpha = build_exchange_matrix(
        density=alpha_matrix,
        electron_repulsion=eri,
    )

    exchange_beta = build_exchange_matrix(
        density=beta_matrix,
        electron_repulsion=eri,
    )

    fock_alpha = (
        core_matrix
        + coulomb
        - exchange_alpha
    )

    fock_beta = (
        core_matrix
        + coulomb
        - exchange_beta
    )

    return (
        np.asarray(
            fock_alpha,
            dtype=np.float64,
        ),
        np.asarray(
            fock_beta,
            dtype=np.float64,
        ),
    )