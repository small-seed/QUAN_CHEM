from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _validate_square_matrix(
    matrix: FloatArray,
    name: str,
) -> FloatArray:
    """
    Validate and return a finite real square matrix.
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


def _validate_symmetric_matrix(
    matrix: FloatArray,
    name: str,
    *,
    atol: float = 1.0e-10,
) -> FloatArray:
    """
    Validate and return a finite real symmetric matrix.
    """
    array = _validate_square_matrix(
        matrix=matrix,
        name=name,
    )

    if not np.allclose(
        array,
        array.T,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            f"The {name} matrix must be symmetric."
        )

    return array


def build_symmetric_orthogonalizer(
    overlap: FloatArray,
    *,
    eigenvalue_threshold: float = 1.0e-10,
) -> FloatArray:
    """
    Build the symmetric orthogonalization matrix X = S^(-1/2).

    For a symmetric positive-definite AO overlap matrix S,

        S = U s U^T

    where U contains the eigenvectors of S and s is the diagonal
    matrix of overlap eigenvalues. The symmetric orthogonalizer is

        X = S^(-1/2)
          = U s^(-1/2) U^T.

    It satisfies

        X^T S X = I.

    Parameters
    ----------
    overlap
        AO overlap matrix S with shape (n_basis, n_basis).
    eigenvalue_threshold
        Smallest allowed overlap eigenvalue. Values at or below this
        threshold indicate linear dependence or severe ill-conditioning
        in the AO basis.

    Returns
    -------
    FloatArray
        Symmetric orthogonalization matrix X with shape
        (n_basis, n_basis).
    """
    overlap_matrix = _validate_symmetric_matrix(
        matrix=overlap,
        name="overlap",
    )

    if eigenvalue_threshold <= 0.0:
        raise ValueError(
            "eigenvalue_threshold must be positive."
        )

    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(
        overlap_matrix
    )

    if np.any(overlap_eigenvalues <= eigenvalue_threshold):
        smallest = float(np.min(overlap_eigenvalues))
        raise ValueError(
            "The overlap matrix is singular or nearly linearly "
            "dependent. "
            f"Smallest eigenvalue = {smallest:.3e}, "
            f"threshold = {eigenvalue_threshold:.3e}."
        )

    inverse_sqrt_eigenvalues = 1.0 / np.sqrt(
        overlap_eigenvalues
    )

    orthogonalizer = (
        overlap_eigenvectors
        @ np.diag(inverse_sqrt_eigenvalues)
        @ overlap_eigenvectors.T
    )

    return orthogonalizer


def solve_roothaan(
    fock: FloatArray,
    overlap: FloatArray,
    *,
    eigenvalue_threshold: float = 1.0e-10,
) -> tuple[FloatArray, FloatArray]:
    """
    Solve the Roothaan-Hall generalized eigenvalue problem

        F C = S C epsilon.

    The AO basis is generally non-orthogonal, so S != I. We first
    construct the symmetric orthogonalizer

        X = S^(-1/2),

    which satisfies

        X^T S X = I.

    With

        C = X C',

    the generalized eigenvalue problem becomes

        F' C' = C' epsilon,

    where

        F' = X^T F X.

    The transformed Fock matrix F' is symmetric and can therefore be
    diagonalized with numpy.linalg.eigh. The AO molecular-orbital
    coefficient matrix is then recovered from

        C = X C'.

    The returned coefficients satisfy, up to numerical precision,

        C^T S C = I.

    Parameters
    ----------
    fock
        AO Fock matrix F with shape (n_basis, n_basis).
    overlap
        AO overlap matrix S with shape (n_basis, n_basis).
    eigenvalue_threshold
        Smallest allowed overlap eigenvalue when constructing S^(-1/2).

    Returns
    -------
    tuple[FloatArray, FloatArray]
        orbital_energies
            One-dimensional array containing the orbital energies
            epsilon_i in ascending order.
        coefficients
            AO molecular-orbital coefficient matrix C. Column i contains
            the AO coefficients of molecular orbital i.
    """
    fock_matrix = _validate_symmetric_matrix(
        matrix=fock,
        name="fock",
    )

    overlap_matrix = _validate_symmetric_matrix(
        matrix=overlap,
        name="overlap",
    )

    if fock_matrix.shape != overlap_matrix.shape:
        raise ValueError(
            "Fock and overlap matrices must have equal shapes."
        )

    orthogonalizer = build_symmetric_orthogonalizer(
        overlap=overlap_matrix,
        eigenvalue_threshold=eigenvalue_threshold,
    )

    transformed_fock = (
        orthogonalizer.T
        @ fock_matrix
        @ orthogonalizer
    )

    # Remove tiny asymmetry introduced by floating-point arithmetic.
    transformed_fock = 0.5 * (
        transformed_fock + transformed_fock.T
    )

    orbital_energies, transformed_coefficients = np.linalg.eigh(
        transformed_fock
    )

    coefficients = (
        orthogonalizer
        @ transformed_coefficients
    )

    return orbital_energies, coefficients