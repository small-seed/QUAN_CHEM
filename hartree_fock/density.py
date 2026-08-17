from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _validate_coefficient_matrix(
    coefficients: FloatArray,
) -> FloatArray:
    """
    Validate and return an MO-coefficient matrix.

    The matrix convention is:

        coefficients[mu, i]

    where mu labels an AO basis function and i labels an MO.
    """
    matrix = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "The MO-coefficient matrix must be two-dimensional."
        )

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            "The MO-coefficient matrix cannot be empty."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "The MO-coefficient matrix must contain "
            "only finite values."
        )

    return matrix


def _validate_occupation(
    n_occupied: int,
    n_orbitals: int,
) -> None:
    """
    Validate the number of occupied orbitals.
    """
    if not isinstance(n_occupied, (int, np.integer)):
        raise TypeError(
            "The number of occupied orbitals must be an integer."
        )

    if n_occupied < 0:
        raise ValueError(
            "The number of occupied orbitals cannot be negative."
        )

    if n_occupied > n_orbitals:
        raise ValueError(
            "The number of occupied orbitals cannot exceed "
            "the number of available molecular orbitals."
        )


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


def build_spin_density(
    coefficients: FloatArray,
    n_occupied: int,
) -> FloatArray:
    """
    Build an AO density matrix for one electron spin.

    Parameters
    ----------
    coefficients
        MO coefficients C[mu, i]. Rows correspond to AO basis
        functions and columns correspond to molecular orbitals.

        Molecular orbitals must be ordered so that the occupied
        orbitals are the first ``n_occupied`` columns.

    n_occupied
        Number of occupied orbitals of this spin.

    Returns
    -------
    density
        One-spin AO density matrix:

            P[mu, nu] =
                sum_i^occupied C[mu, i] C[nu, i].

    Notes
    -----
    No factor of two is included. This function is therefore the
    basic density builder for one UHF spin channel.
    """
    coefficient_matrix = (
        _validate_coefficient_matrix(
            coefficients
        )
    )

    _validate_occupation(
        n_occupied=n_occupied,
        n_orbitals=coefficient_matrix.shape[1],
    )

    occupied_coefficients = (
        coefficient_matrix[:, :n_occupied]
    )

    density = (
        occupied_coefficients
        @ occupied_coefficients.T
    )

    return np.asarray(
        density,
        dtype=np.float64,
    )


def build_rhf_density(
    coefficients: FloatArray,
    n_occupied: int,
) -> FloatArray:
    """
    Build the spin-summed RHF AO density matrix.

    Each occupied spatial molecular orbital contains one alpha and
    one beta electron. Therefore:

        P[mu, nu] =
            2 sum_i^occupied C[mu, i] C[nu, i].

    Parameters
    ----------
    coefficients
        RHF MO coefficients C[mu, i]. Occupied orbitals must be the
        first ``n_occupied`` columns.

    n_occupied
        Number of doubly occupied spatial molecular orbitals:

            n_occupied = n_electrons // 2.
    """
    return (
        2.0
        * build_spin_density(
            coefficients=coefficients,
            n_occupied=n_occupied,
        )
    )


def build_uhf_density(
    coefficients_alpha: FloatArray,
    coefficients_beta: FloatArray,
    n_alpha: int,
    n_beta: int,
) -> tuple[FloatArray, FloatArray]:
    """
    Build the alpha- and beta-spin UHF AO density matrices.

    The two returned matrices are:

        P_alpha[mu, nu] =
            sum_i^n_alpha C_alpha[mu, i] C_alpha[nu, i]

        P_beta[mu, nu] =
            sum_i^n_beta C_beta[mu, i] C_beta[nu, i].

    No factor of two is used because each spin orbital contains one
    electron.
    """
    alpha_matrix = (
        _validate_coefficient_matrix(
            coefficients_alpha
        )
    )

    beta_matrix = (
        _validate_coefficient_matrix(
            coefficients_beta
        )
    )

    if alpha_matrix.shape[0] != beta_matrix.shape[0]:
        raise ValueError(
            "Alpha and beta coefficient matrices must use "
            "the same number of AO basis functions."
        )

    density_alpha = build_spin_density(
        coefficients=alpha_matrix,
        n_occupied=n_alpha,
    )

    density_beta = build_spin_density(
        coefficients=beta_matrix,
        n_occupied=n_beta,
    )

    return density_alpha, density_beta


def density_electron_count(
    density: FloatArray,
    overlap: FloatArray,
) -> float:
    """
    Recover the electron count from an AO density matrix.

    AO basis functions are generally non-orthogonal, so the electron
    count is not simply Tr(P). It is:

        N = Tr(P S)
          = sum_mu,nu P[mu, nu] S[nu, mu].

    For an RHF density this returns the total electron count. For a
    one-spin UHF density it returns either n_alpha or n_beta.
    """
    density_matrix = _validate_square_matrix(
        matrix=density,
        name="density",
    )

    overlap_matrix = _validate_square_matrix(
        matrix=overlap,
        name="overlap",
    )

    if density_matrix.shape != overlap_matrix.shape:
        raise ValueError(
            "Density and overlap matrices must have equal shapes."
        )

    return float(
        np.einsum(
            "mn,nm->",
            density_matrix,
            overlap_matrix,
            optimize=True,
        )
    )