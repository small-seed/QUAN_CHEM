from __future__ import annotations

from functools import lru_cache
from math import erf, exp, factorial, pi, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from common.basis import BasisSet
from common.gaussian import ContractedGaussian
from common.molecule import Molecule


FloatArray = NDArray[np.float64]
AngularMomentum = tuple[int, int, int]
Vector3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Small data-access helpers
# ---------------------------------------------------------------------------

def _as_vector3(
    value: FloatArray,
) -> Vector3:
    return (
        float(value[0]),
        float(value[1]),
        float(value[2]),
    )

# ---------------------------------------------------------------------------
# Boys function
# ---------------------------------------------------------------------------

def boys_function(
    order: int,
    argument: float,
) -> float:
    """
    Evaluate the Boys function

        F_n(T) = integral_0^1 t^(2n) exp(-T t^2) dt.

    A power series is used near T = 0 and upward recursion elsewhere.
    """
    if order < 0:
        raise ValueError(
            "Boys-function order must be non-negative."
        )

    if not np.isfinite(argument) or argument < 0.0:
        raise ValueError(
            "Boys-function argument must be finite "
            "and non-negative."
        )

    if argument < 1.0e-10:
        result = 0.0
        term = 1.0

        for k in range(100):
            contribution = (
                term
                / (
                    factorial(k)
                    * (2 * order + 2 * k + 1)
                )
            )

            result += contribution

            if abs(contribution) < 1.0e-16:
                break

            term *= -argument

        return result

    root = sqrt(argument)

    value = (
        0.5
        * sqrt(pi / argument)
        * erf(root)
    )

    if order == 0:
        return value

    exponential = exp(-argument)

    for n in range(order):
        value = (
            (2 * n + 1) * value
            - exponential
        ) / (2.0 * argument)

    return value


# ---------------------------------------------------------------------------
# Hermite Gaussian expansion
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _hermite_coefficient(
    i: int,
    j: int,
    t: int,
    displacement: float,
    alpha: float,
    beta: float,
) -> float:
    """
    Hermite expansion coefficient E_t^(i,j).

    The Gaussian-product factor exp(-q Q^2) is contained in E_0^(0,0).
    """
    if (
        i < 0
        or j < 0
        or t < 0
        or t > i + j
    ):
        return 0.0

    p = alpha + beta
    q = alpha * beta / p

    if i == 0 and j == 0 and t == 0:
        return exp(
            -q * displacement * displacement
        )

    if j == 0:
        return (
            _hermite_coefficient(
                i - 1,
                j,
                t - 1,
                displacement,
                alpha,
                beta,
            )
            / (2.0 * p)
            - (
                q
                * displacement
                / alpha
            )
            * _hermite_coefficient(
                i - 1,
                j,
                t,
                displacement,
                alpha,
                beta,
            )
            + (t + 1)
            * _hermite_coefficient(
                i - 1,
                j,
                t + 1,
                displacement,
                alpha,
                beta,
            )
        )

    return (
        _hermite_coefficient(
            i,
            j - 1,
            t - 1,
            displacement,
            alpha,
            beta,
        )
        / (2.0 * p)
        + (
            q
            * displacement
            / beta
        )
        * _hermite_coefficient(
            i,
            j - 1,
            t,
            displacement,
            alpha,
            beta,
        )
        + (t + 1)
        * _hermite_coefficient(
            i,
            j - 1,
            t + 1,
            displacement,
            alpha,
            beta,
        )
    )


def _gaussian_product_center(
    alpha: float,
    center_a: Vector3,
    beta: float,
    center_b: Vector3,
) -> Vector3:
    p = alpha + beta

    return (
        (
            alpha * center_a[0]
            + beta * center_b[0]
        ) / p,
        (
            alpha * center_a[1]
            + beta * center_b[1]
        ) / p,
        (
            alpha * center_a[2]
            + beta * center_b[2]
        ) / p,
    )


@lru_cache(maxsize=None)
def _coulomb_auxiliary(
    t: int,
    u: int,
    v: int,
    order: int,
    exponent: float,
    pc_x: float,
    pc_y: float,
    pc_z: float,
) -> float:
    """
    Hermite Coulomb auxiliary integral R_tuv^n.
    """
    if t < 0 or u < 0 or v < 0:
        return 0.0

    if t == 0 and u == 0 and v == 0:
        argument = exponent * (
            pc_x * pc_x
            + pc_y * pc_y
            + pc_z * pc_z
        )

        return (
            (-2.0 * exponent) ** order
            * boys_function(
                order,
                argument,
            )
        )

    if t > 0:
        value = (
            pc_x
            * _coulomb_auxiliary(
                t - 1,
                u,
                v,
                order + 1,
                exponent,
                pc_x,
                pc_y,
                pc_z,
            )
        )

        if t > 1:
            value += (
                (t - 1)
                * _coulomb_auxiliary(
                    t - 2,
                    u,
                    v,
                    order + 1,
                    exponent,
                    pc_x,
                    pc_y,
                    pc_z,
                )
            )

        return value

    if u > 0:
        value = (
            pc_y
            * _coulomb_auxiliary(
                t,
                u - 1,
                v,
                order + 1,
                exponent,
                pc_x,
                pc_y,
                pc_z,
            )
        )

        if u > 1:
            value += (
                (u - 1)
                * _coulomb_auxiliary(
                    t,
                    u - 2,
                    v,
                    order + 1,
                    exponent,
                    pc_x,
                    pc_y,
                    pc_z,
                )
            )

        return value

    value = (
        pc_z
        * _coulomb_auxiliary(
            t,
            u,
            v - 1,
            order + 1,
            exponent,
            pc_x,
            pc_y,
            pc_z,
        )
    )

    if v > 1:
        value += (
            (v - 1)
            * _coulomb_auxiliary(
                t,
                u,
                v - 2,
                order + 1,
                exponent,
                pc_x,
                pc_y,
                pc_z,
            )
        )

    return value


# ---------------------------------------------------------------------------
# Primitive one-electron integrals
# ---------------------------------------------------------------------------

def primitive_overlap(
    alpha: float,
    angular_a: AngularMomentum,
    center_a: Vector3,
    beta: float,
    angular_b: AngularMomentum,
    center_b: Vector3,
) -> float:
    """
    Unnormalized primitive Cartesian-Gaussian overlap integral.
    """
    p = alpha + beta

    ex = _hermite_coefficient(
        angular_a[0],
        angular_b[0],
        0,
        center_a[0] - center_b[0],
        alpha,
        beta,
    )
    ey = _hermite_coefficient(
        angular_a[1],
        angular_b[1],
        0,
        center_a[1] - center_b[1],
        alpha,
        beta,
    )
    ez = _hermite_coefficient(
        angular_a[2],
        angular_b[2],
        0,
        center_a[2] - center_b[2],
        alpha,
        beta,
    )

    return (
        ex
        * ey
        * ez
        * (pi / p) ** 1.5
    )


def primitive_kinetic(
    alpha: float,
    angular_a: AngularMomentum,
    center_a: Vector3,
    beta: float,
    angular_b: AngularMomentum,
    center_b: Vector3,
) -> float:
    """
    Unnormalized primitive kinetic-energy integral

        <a| -1/2 nabla^2 |b>.
    """
    l_b, m_b, n_b = angular_b

    base = primitive_overlap(
        alpha,
        angular_a,
        center_a,
        beta,
        angular_b,
        center_b,
    )

    raised = (
        primitive_overlap(
            alpha,
            angular_a,
            center_a,
            beta,
            (l_b + 2, m_b, n_b),
            center_b,
        )
        + primitive_overlap(
            alpha,
            angular_a,
            center_a,
            beta,
            (l_b, m_b + 2, n_b),
            center_b,
        )
        + primitive_overlap(
            alpha,
            angular_a,
            center_a,
            beta,
            (l_b, m_b, n_b + 2),
            center_b,
        )
    )

    lowered = 0.0

    if l_b >= 2:
        lowered += (
            l_b
            * (l_b - 1)
            * primitive_overlap(
                alpha,
                angular_a,
                center_a,
                beta,
                (l_b - 2, m_b, n_b),
                center_b,
            )
        )

    if m_b >= 2:
        lowered += (
            m_b
            * (m_b - 1)
            * primitive_overlap(
                alpha,
                angular_a,
                center_a,
                beta,
                (l_b, m_b - 2, n_b),
                center_b,
            )
        )

    if n_b >= 2:
        lowered += (
            n_b
            * (n_b - 1)
            * primitive_overlap(
                alpha,
                angular_a,
                center_a,
                beta,
                (l_b, m_b, n_b - 2),
                center_b,
            )
        )

    return (
        beta
        * (
            2 * (l_b + m_b + n_b)
            + 3
        )
        * base
        - 2.0
        * beta
        * beta
        * raised
        - 0.5
        * lowered
    )


def primitive_nuclear_attraction(
    alpha: float,
    angular_a: AngularMomentum,
    center_a: Vector3,
    beta: float,
    angular_b: AngularMomentum,
    center_b: Vector3,
    nucleus_center: Vector3,
    nuclear_charge: float,
) -> float:
    """
    Unnormalized primitive electron-nucleus attraction integral.
    """
    p = alpha + beta

    product_center = _gaussian_product_center(
        alpha,
        center_a,
        beta,
        center_b,
    )

    pc = (
        product_center[0] - nucleus_center[0],
        product_center[1] - nucleus_center[1],
        product_center[2] - nucleus_center[2],
    )

    result = 0.0

    for t in range(
        angular_a[0] + angular_b[0] + 1
    ):
        ex = _hermite_coefficient(
            angular_a[0],
            angular_b[0],
            t,
            center_a[0] - center_b[0],
            alpha,
            beta,
        )

        for u in range(
            angular_a[1] + angular_b[1] + 1
        ):
            ey = _hermite_coefficient(
                angular_a[1],
                angular_b[1],
                u,
                center_a[1] - center_b[1],
                alpha,
                beta,
            )

            for v in range(
                angular_a[2] + angular_b[2] + 1
            ):
                ez = _hermite_coefficient(
                    angular_a[2],
                    angular_b[2],
                    v,
                    center_a[2] - center_b[2],
                    alpha,
                    beta,
                )

                result += (
                    ex
                    * ey
                    * ez
                    * _coulomb_auxiliary(
                        t,
                        u,
                        v,
                        0,
                        p,
                        pc[0],
                        pc[1],
                        pc[2],
                    )
                )

    return (
        -nuclear_charge
        * 2.0
        * pi
        / p
        * result
    )


# ---------------------------------------------------------------------------
# Primitive two-electron integral
# ---------------------------------------------------------------------------

def primitive_electron_repulsion(
    alpha: float,
    angular_a: AngularMomentum,
    center_a: Vector3,
    beta: float,
    angular_b: AngularMomentum,
    center_b: Vector3,
    gamma: float,
    angular_c: AngularMomentum,
    center_c: Vector3,
    delta: float,
    angular_d: AngularMomentum,
    center_d: Vector3,
) -> float:
    """
    Unnormalized primitive electron-repulsion integral

        (ab|cd).
    """
    p = alpha + beta
    q = gamma + delta

    product_p = _gaussian_product_center(
        alpha,
        center_a,
        beta,
        center_b,
    )
    product_q = _gaussian_product_center(
        gamma,
        center_c,
        delta,
        center_d,
    )

    reduced_exponent = p * q / (p + q)

    pq = (
        product_p[0] - product_q[0],
        product_p[1] - product_q[1],
        product_p[2] - product_q[2],
    )

    result = 0.0

    for t in range(
        angular_a[0] + angular_b[0] + 1
    ):
        ex_ab = _hermite_coefficient(
            angular_a[0],
            angular_b[0],
            t,
            center_a[0] - center_b[0],
            alpha,
            beta,
        )

        for u in range(
            angular_a[1] + angular_b[1] + 1
        ):
            ey_ab = _hermite_coefficient(
                angular_a[1],
                angular_b[1],
                u,
                center_a[1] - center_b[1],
                alpha,
                beta,
            )

            for v in range(
                angular_a[2] + angular_b[2] + 1
            ):
                ez_ab = _hermite_coefficient(
                    angular_a[2],
                    angular_b[2],
                    v,
                    center_a[2] - center_b[2],
                    alpha,
                    beta,
                )

                for tau in range(
                    angular_c[0] + angular_d[0] + 1
                ):
                    ex_cd = _hermite_coefficient(
                        angular_c[0],
                        angular_d[0],
                        tau,
                        center_c[0] - center_d[0],
                        gamma,
                        delta,
                    )

                    for phi in range(
                        angular_c[1] + angular_d[1] + 1
                    ):
                        ey_cd = _hermite_coefficient(
                            angular_c[1],
                            angular_d[1],
                            phi,
                            center_c[1] - center_d[1],
                            gamma,
                            delta,
                        )

                        for chi in range(
                            angular_c[2] + angular_d[2] + 1
                        ):
                            ez_cd = _hermite_coefficient(
                                angular_c[2],
                                angular_d[2],
                                chi,
                                center_c[2] - center_d[2],
                                gamma,
                                delta,
                            )

                            phase = (
                                -1.0
                                if (
                                    tau
                                    + phi
                                    + chi
                                ) % 2
                                else 1.0
                            )

                            result += (
                                ex_ab
                                * ey_ab
                                * ez_ab
                                * ex_cd
                                * ey_cd
                                * ez_cd
                                * phase
                                * _coulomb_auxiliary(
                                    t + tau,
                                    u + phi,
                                    v + chi,
                                    0,
                                    reduced_exponent,
                                    pq[0],
                                    pq[1],
                                    pq[2],
                                )
                            )

    prefactor = (
        2.0
        * pi ** 2.5
        / (
            p
            * q
            * sqrt(p + q)
        )
    )

    return prefactor * result


# ---------------------------------------------------------------------------
# Contracted integrals
# ---------------------------------------------------------------------------

def overlap_integral(
    function_a: ContractedGaussian,
    function_b: ContractedGaussian,
) -> float:
    center_a = _as_vector3(function_a.center)
    center_b = _as_vector3(function_b.center)

    angular_a = function_a.angular_momentum
    angular_b = function_b.angular_momentum

    total = 0.0

    for primitive_a in function_a.primitives:
        alpha = primitive_a.exponent

        weight_a = (
            primitive_a.coefficient
            * primitive_a.normalization
        )

        for primitive_b in function_b.primitives:
            beta = primitive_b.exponent

            weight_b = (
                primitive_b.coefficient
                * primitive_b.normalization
            )

            total += (
                weight_a
                * weight_b
                * primitive_overlap(
                    alpha,
                    angular_a,
                    center_a,
                    beta,
                    angular_b,
                    center_b,
                )
            )

    return (
        function_a.contraction_normalization
        * function_b.contraction_normalization
        * total
    )


def kinetic_integral(
    function_a: ContractedGaussian,
    function_b: ContractedGaussian,
) -> float:
    center_a = _as_vector3(function_a.center)
    center_b = _as_vector3(function_b.center)

    angular_a = function_a.angular_momentum
    angular_b = function_b.angular_momentum

    total = 0.0

    for primitive_a in function_a.primitives:
        alpha = primitive_a.exponent

        weight_a = (
            primitive_a.coefficient
            * primitive_a.normalization
        )

        for primitive_b in function_b.primitives:
            beta = primitive_b.exponent

            weight_b = (
                primitive_b.coefficient
                * primitive_b.normalization
            )

            total += (
                weight_a
                * weight_b
                * primitive_kinetic(
                    alpha,
                    angular_a,
                    center_a,
                    beta,
                    angular_b,
                    center_b,
                )
            )

    return (
        function_a.contraction_normalization
        * function_b.contraction_normalization
        * total
    )


def nuclear_attraction_integral(
    function_a: ContractedGaussian,
    function_b: ContractedGaussian,
    nuclei: Iterable[
        tuple[float, Vector3]
    ],
) -> float:
    center_a = _as_vector3(function_a.center)
    center_b = _as_vector3(function_b.center)

    angular_a = function_a.angular_momentum
    angular_b = function_b.angular_momentum

    total = 0.0

    for primitive_a in function_a.primitives:
        alpha = primitive_a.exponent

        weight_a = (
            primitive_a.coefficient
            * primitive_a.normalization
        )

        for primitive_b in function_b.primitives:
            beta = primitive_b.exponent

            weight_b = (
                primitive_b.coefficient
                * primitive_b.normalization
            )

            primitive_sum = 0.0

            for charge, center in nuclei:
                primitive_sum += (
                    primitive_nuclear_attraction(
                        alpha,
                        angular_a,
                        center_a,
                        beta,
                        angular_b,
                        center_b,
                        center,
                        charge,
                    )
                )

            total += (
                weight_a
                * weight_b
                * primitive_sum
            )

    return (
        function_a.contraction_normalization
        * function_b.contraction_normalization
        * total
    )


def electron_repulsion_integral(
    function_a: ContractedGaussian,
    function_b: ContractedGaussian,
    function_c: ContractedGaussian,
    function_d: ContractedGaussian,
) -> float:
    centers = (
        _as_vector3(function_a.center),
        _as_vector3(function_b.center),
        _as_vector3(function_c.center),
        _as_vector3(function_d.center),
    )

    angular = (
        function_a.angular_momentum,
        function_b.angular_momentum,
        function_c.angular_momentum,
        function_d.angular_momentum,
    )

    total = 0.0

    for primitive_a in function_a.primitives:
        alpha = primitive_a.exponent
        weight_a = (
            primitive_a.coefficient
            * primitive_a.normalization
        )

        for primitive_b in function_b.primitives:
            beta = primitive_b.exponent
            weight_b = (
                primitive_b.coefficient
                * primitive_b.normalization
            )

            for primitive_c in function_c.primitives:
                gamma = primitive_c.exponent
                weight_c = (
                    primitive_c.coefficient
                    * primitive_c.normalization
                )

                for primitive_d in function_d.primitives:
                    delta = primitive_d.exponent
                    weight_d = (
                        primitive_d.coefficient
                        * primitive_d.normalization
                    )

                    total += (
                        weight_a
                        * weight_b
                        * weight_c
                        * weight_d
                        * primitive_electron_repulsion(
                            alpha,
                            angular[0],
                            centers[0],
                            beta,
                            angular[1],
                            centers[1],
                            gamma,
                            angular[2],
                            centers[2],
                            delta,
                            angular[3],
                            centers[3],
                        )
                    )

    return (
        function_a.contraction_normalization
        * function_b.contraction_normalization
        * function_c.contraction_normalization
        * function_d.contraction_normalization
        * total
    )


# ---------------------------------------------------------------------------
# Molecule access
# ---------------------------------------------------------------------------

def _molecule_nuclei(
    molecule: Molecule,
) -> tuple[tuple[float, Vector3], ...]:
    return tuple(
        (
            float(atom.atomic_number),
            _as_vector3(atom.position),
        )
        for atom in molecule.atoms
    )


# ---------------------------------------------------------------------------
# Matrix and tensor builders
# ---------------------------------------------------------------------------

def build_overlap_matrix(
    basis: BasisSet,
) -> FloatArray:
    functions = basis.functions
    n_functions = len(functions)

    matrix = np.zeros(
        (n_functions, n_functions),
        dtype=float,
    )

    for mu in range(n_functions):
        for nu in range(mu + 1):
            value = overlap_integral(
                functions[mu],
                functions[nu],
            )

            matrix[mu, nu] = value
            matrix[nu, mu] = value

    return matrix


def build_kinetic_matrix(
    basis: BasisSet,
) -> FloatArray:
    functions = basis.functions
    n_functions = len(functions)

    matrix = np.zeros(
        (n_functions, n_functions),
        dtype=float,
    )

    for mu in range(n_functions):
        for nu in range(mu + 1):
            value = kinetic_integral(
                functions[mu],
                functions[nu],
            )

            matrix[mu, nu] = value
            matrix[nu, mu] = value

    return matrix


def build_nuclear_attraction_matrix(
    molecule: Molecule,
    basis: BasisSet,
) -> FloatArray:
    functions = basis.functions
    nuclei = _molecule_nuclei(molecule)

    n_functions = len(functions)

    matrix = np.zeros(
        (n_functions, n_functions),
        dtype=float,
    )

    for mu in range(n_functions):
        for nu in range(mu + 1):
            value = nuclear_attraction_integral(
                functions[mu],
                functions[nu],
                nuclei,
            )

            matrix[mu, nu] = value
            matrix[nu, mu] = value

    return matrix


def build_core_hamiltonian(
    molecule: Molecule,
    basis: BasisSet,
) -> FloatArray:
    return (
        build_kinetic_matrix(basis)
        + build_nuclear_attraction_matrix(
            molecule,
            basis,
        )
    )


def build_eri_tensor(
    basis: BasisSet,
) -> FloatArray:
    """
    Build the chemists' notation ERI tensor

        eri[mu, nu, lam, sig] = (mu nu | lam sig).

    Eightfold permutation symmetry is used to avoid recomputing
    equivalent integrals.
    """
    functions = basis.functions
    n_functions = len(functions)

    eri = np.zeros(
        (
            n_functions,
            n_functions,
            n_functions,
            n_functions,
        ),
        dtype=float,
    )

    pairs = [
        (mu, nu)
        for mu in range(n_functions)
        for nu in range(mu + 1)
    ]

    for pair_ab_index, (mu, nu) in enumerate(
        pairs
    ):
        for pair_cd_index in range(
            pair_ab_index + 1
        ):
            lam, sig = pairs[pair_cd_index]

            value = electron_repulsion_integral(
                functions[mu],
                functions[nu],
                functions[lam],
                functions[sig],
            )

            permutations = {
                (mu, nu, lam, sig),
                (nu, mu, lam, sig),
                (mu, nu, sig, lam),
                (nu, mu, sig, lam),
                (lam, sig, mu, nu),
                (sig, lam, mu, nu),
                (lam, sig, nu, mu),
                (sig, lam, nu, mu),
            }

            for indices in permutations:
                eri[indices] = value

    return eri


def build_integrals(
    molecule: Molecule,
    basis: BasisSet,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """
    Build all AO quantities required by a basic Hartree-Fock solver.

    Returns
    -------
    overlap
        S[mu, nu].

    kinetic
        T[mu, nu].

    nuclear_attraction
        V[mu, nu].

    core_hamiltonian
        H_core = T + V.

    electron_repulsion
        (mu nu | lambda sigma).
    """
    overlap = build_overlap_matrix(basis)
    kinetic = build_kinetic_matrix(basis)

    nuclear_attraction = (
        build_nuclear_attraction_matrix(
            molecule,
            basis,
        )
    )

    core_hamiltonian = (
        kinetic
        + nuclear_attraction
    )

    electron_repulsion = build_eri_tensor(
        basis
    )

    return (
        overlap,
        kinetic,
        nuclear_attraction,
        core_hamiltonian,
        electron_repulsion,
    )