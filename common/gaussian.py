from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
AngularMomentum = tuple[int, int, int]


def double_factorial(n: int) -> int:
    """
    Compute n!!.

    Conventions:
        (-1)!! = 1
         0!!   = 1
    """
    if n <= 0:
        return 1

    result = 1

    for value in range(n, 0, -2):
        result *= value

    return result


def primitive_normalization(
    exponent: float,
    angular_momentum: AngularMomentum,
) -> float:
    """
    Return the normalization constant of a Cartesian
    primitive Gaussian:

        g(r) =
            N
            (x - Ax)^l
            (y - Ay)^m
            (z - Az)^n
            exp[-alpha |r - A|^2]

    such that:

        integral |g(r)|^2 dr = 1
    """
    if (
        not np.isfinite(exponent)
        or exponent <= 0.0
    ):
        raise ValueError(
            "Gaussian exponent must be a finite "
            "positive number."
        )

    if len(angular_momentum) != 3:
        raise ValueError(
            "Angular momentum must be a tuple (l, m, n)."
        )

    if any(
        not isinstance(value, int)
        for value in angular_momentum
    ):
        raise TypeError(
            "Angular momentum values must be integers."
        )

    if any(
        value < 0
        for value in angular_momentum
    ):
        raise ValueError(
            "Angular momentum values must be non-negative."
        )

    l, m, n = angular_momentum

    denominator = (
        double_factorial(2 * l - 1)
        * double_factorial(2 * m - 1)
        * double_factorial(2 * n - 1)
    )

    angular_factor = (
        (4.0 * exponent) ** (l + m + n)
        / denominator
    )

    return float(
        (2.0 * exponent / pi) ** 0.75
        * sqrt(angular_factor)
    )


@dataclass(frozen=True)
class PrimitiveGaussian:
    """
    Cartesian primitive Gaussian contribution:

        d_p N_p
        (x - Ax)^l
        (y - Ay)^m
        (z - Az)^n
        exp[-alpha_p |r - A|^2]

    Parameters
    ----------
    exponent
        Gaussian exponent alpha_p.

    coefficient
        Contraction coefficient d_p.

    center
        Gaussian center A = (Ax, Ay, Az), in bohr.

    angular_momentum
        Cartesian angular momentum tuple (l, m, n).

        Examples:
            (0, 0, 0) -> s
            (1, 0, 0) -> px
            (0, 1, 0) -> py
            (0, 0, 1) -> pz
    """

    exponent: float
    coefficient: float
    center: FloatArray
    angular_momentum: AngularMomentum = (0, 0, 0)

    def __post_init__(self) -> None:
        center = np.asarray(
            self.center,
            dtype=np.float64,
        ).copy()

        if center.shape != (3,):
            raise ValueError(
                "Gaussian center must have shape (3,)."
            )

        if not np.all(np.isfinite(center)):
            raise ValueError(
                "Gaussian center must contain finite coordinates."
            )

        if (
            not np.isfinite(self.exponent)
            or self.exponent <= 0.0
        ):
            raise ValueError(
                "Gaussian exponent must be a finite "
                "positive number."
            )

        if not np.isfinite(self.coefficient):
            raise ValueError(
                "Gaussian coefficient must be finite."
            )

        if len(self.angular_momentum) != 3:
            raise ValueError(
                "Angular momentum must be a tuple (l, m, n)."
            )

        if any(
            not isinstance(value, int)
            for value in self.angular_momentum
        ):
            raise TypeError(
                "Angular momentum values must be integers."
            )

        if any(
            value < 0
            for value in self.angular_momentum
        ):
            raise ValueError(
                "Angular momentum values must be non-negative."
            )

        center.setflags(write=False)

        object.__setattr__(
            self,
            "center",
            center,
        )

    @property
    def l(self) -> int:
        return self.angular_momentum[0]

    @property
    def m(self) -> int:
        return self.angular_momentum[1]

    @property
    def n(self) -> int:
        return self.angular_momentum[2]

    @property
    def total_angular_momentum(self) -> int:
        """
        Total Cartesian angular momentum:

            L = l + m + n
        """
        return self.l + self.m + self.n

    @property
    def normalization(self) -> float:
        """
        Primitive normalization constant N_p.
        """
        return primitive_normalization(
            exponent=self.exponent,
            angular_momentum=self.angular_momentum,
        )

    def unnormalized_value(
        self,
        point: FloatArray,
    ) -> float:
        """
        Evaluate the primitive Gaussian without normalization
        and without the contraction coefficient:

            (x - Ax)^l
            (y - Ay)^m
            (z - Az)^n
            exp[-alpha |r - A|^2]
        """
        point_array = np.asarray(
            point,
            dtype=np.float64,
        )

        if point_array.shape != (3,):
            raise ValueError(
                "Evaluation point must have shape (3,)."
            )

        if not np.all(np.isfinite(point_array)):
            raise ValueError(
                "Evaluation point must contain finite coordinates."
            )

        displacement = point_array - self.center

        x, y, z = displacement

        polynomial = (
            x**self.l
            * y**self.m
            * z**self.n
        )

        squared_distance = float(
            np.dot(
                displacement,
                displacement,
            )
        )

        radial_part = np.exp(
            -self.exponent * squared_distance
        )

        return float(
            polynomial * radial_part
        )

    def normalized_value(
        self,
        point: FloatArray,
    ) -> float:
        """
        Evaluate the normalized primitive Gaussian,
        excluding the contraction coefficient:

            N_p g_p(r)
        """
        return (
            self.normalization
            * self.unnormalized_value(point)
        )

    def value(
        self,
        point: FloatArray,
    ) -> float:
        """
        Evaluate the primitive contribution to a contraction:

            d_p N_p g_p(r)
        """
        return (
            self.coefficient
            * self.normalized_value(point)
        )

    def __call__(
        self,
        point: FloatArray,
    ) -> float:
        return self.value(point)


def normalized_primitive_overlap_same_center(
    primitive_a: PrimitiveGaussian,
    primitive_b: PrimitiveGaussian,
) -> float:
    """
    Compute the overlap between two normalized Cartesian
    primitive Gaussians with the same center and the same
    angular momentum.

    Contraction coefficients are not included.

    For normalized primitives:

        S_ab =
            integral g_a(r) g_b(r) dr

            =
            [
                2 sqrt(alpha_a alpha_b)
                ------------------------
                   alpha_a + alpha_b
            ]^(l + m + n + 3/2)
    """
    if not np.allclose(
        primitive_a.center,
        primitive_b.center,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "This overlap function requires "
            "the same Gaussian center."
        )

    if (
        primitive_a.angular_momentum
        != primitive_b.angular_momentum
    ):
        raise ValueError(
            "This overlap function requires "
            "the same angular momentum."
        )

    alpha_a = primitive_a.exponent
    alpha_b = primitive_b.exponent

    total_angular_momentum = (
        primitive_a.total_angular_momentum
    )

    exponent_ratio = (
        2.0 * sqrt(alpha_a * alpha_b)
        / (alpha_a + alpha_b)
    )

    overlap = exponent_ratio ** (
        total_angular_momentum + 1.5
    )

    return float(overlap)


def compute_contraction_normalization(
    primitives: Iterable[PrimitiveGaussian],
) -> float:
    """
    Compute the normalization constant N_c of a contracted
    Cartesian Gaussian.

    The raw contraction is:

        phi_raw(r) =
            sum_p d_p g_p(r)

    where each g_p is individually normalized.

    The normalized contraction is:

        phi(r) =
            N_c phi_raw(r)

    with:

        integral |phi(r)|^2 dr = 1

    Therefore:

        N_c =
            1 / sqrt(
                sum_p sum_q d_p d_q S_pq
            )

    where:

        S_pq =
            integral g_p(r) g_q(r) dr
    """
    primitive_values = tuple(primitives)

    if not primitive_values:
        raise ValueError(
            "At least one primitive is required "
            "to compute contraction normalization."
        )

    reference_center = primitive_values[0].center
    reference_angular_momentum = (
        primitive_values[0].angular_momentum
    )

    for primitive in primitive_values[1:]:
        if not np.allclose(
            primitive.center,
            reference_center,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "All primitives in a contraction "
                "must have the same center."
            )

        if (
            primitive.angular_momentum
            != reference_angular_momentum
        ):
            raise ValueError(
                "All primitives in a contraction "
                "must have the same angular momentum."
            )

    squared_norm = 0.0

    for primitive_p in primitive_values:
        for primitive_q in primitive_values:
            overlap_pq = (
                normalized_primitive_overlap_same_center(
                    primitive_p,
                    primitive_q,
                )
            )

            squared_norm += (
                primitive_p.coefficient
                * primitive_q.coefficient
                * overlap_pq
            )

    if not np.isfinite(squared_norm):
        raise ValueError(
            "Contraction squared norm must be finite."
        )

    if squared_norm <= 0.0:
        raise ValueError(
            "Contraction squared norm must be positive."
        )

    return float(
        1.0 / sqrt(squared_norm)
    )


@dataclass(frozen=True)
class ContractedGaussian:
    """
    Contracted Cartesian Gaussian basis function:

        phi(r) =
            N_c sum_p d_p N_p g_p(r)

    where:
        d_p : contraction coefficient
        N_p : primitive normalization
        N_c : contraction normalization

    All primitives must have:
        - the same center;
        - the same angular momentum.
    """

    primitives: tuple[PrimitiveGaussian, ...]
    contraction_normalization: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        primitives = tuple(self.primitives)

        if not primitives:
            raise ValueError(
                "A contracted Gaussian must contain "
                "at least one primitive."
            )

        if not np.isfinite(
            self.contraction_normalization
        ):
            raise ValueError(
                "Contraction normalization must be finite."
            )

        if self.contraction_normalization <= 0.0:
            raise ValueError(
                "Contraction normalization must be positive."
            )

        reference_center = primitives[0].center
        reference_angular_momentum = (
            primitives[0].angular_momentum
        )

        for primitive in primitives[1:]:
            if not np.allclose(
                primitive.center,
                reference_center,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "All primitives in a contraction "
                    "must have the same center."
                )

            if (
                primitive.angular_momentum
                != reference_angular_momentum
            ):
                raise ValueError(
                    "All primitives in a contraction "
                    "must have the same angular momentum."
                )

        object.__setattr__(
            self,
            "primitives",
            primitives,
        )

    @classmethod
    def from_parameters(
        cls,
        exponents: Iterable[float],
        coefficients: Iterable[float],
        center: FloatArray,
        angular_momentum: AngularMomentum = (0, 0, 0),
        contraction_normalization: float | None = None,
        label: str | None = None,
    ) -> ContractedGaussian:
        """
        Construct a contracted Gaussian from exponent
        and coefficient arrays.

        If contraction_normalization is None,
        it is computed automatically.

        Example
        -------
        hydrogen_1s = ContractedGaussian.from_parameters(
            exponents=[
                3.42525091,
                0.62391373,
                0.16885540,
            ],
            coefficients=[
                0.15432897,
                0.53532814,
                0.44463454,
            ],
            center=np.array([0.0, 0.0, 0.0]),
            angular_momentum=(0, 0, 0),
            label="H STO-3G 1s",
        )
        """
        exponent_values = tuple(exponents)
        coefficient_values = tuple(coefficients)

        if (
            len(exponent_values)
            != len(coefficient_values)
        ):
            raise ValueError(
                "Exponents and coefficients "
                "must have equal lengths."
            )

        if not exponent_values:
            raise ValueError(
                "At least one exponent and coefficient "
                "are required."
            )

        center_array = np.asarray(
            center,
            dtype=np.float64,
        )

        primitives = tuple(
            PrimitiveGaussian(
                exponent=exponent,
                coefficient=coefficient,
                center=center_array.copy(),
                angular_momentum=angular_momentum,
            )
            for exponent, coefficient in zip(
                exponent_values,
                coefficient_values,
                strict=True,
            )
        )

        if contraction_normalization is None:
            contraction_normalization = (
                compute_contraction_normalization(
                    primitives
                )
            )

        return cls(
            primitives=primitives,
            contraction_normalization=(
                contraction_normalization
            ),
            label=label,
        )

    @property
    def center(self) -> FloatArray:
        return self.primitives[0].center

    @property
    def angular_momentum(
        self,
    ) -> AngularMomentum:
        return self.primitives[0].angular_momentum

    @property
    def l(self) -> int:
        return self.angular_momentum[0]

    @property
    def m(self) -> int:
        return self.angular_momentum[1]

    @property
    def n(self) -> int:
        return self.angular_momentum[2]

    @property
    def total_angular_momentum(self) -> int:
        return self.l + self.m + self.n

    @property
    def n_primitives(self) -> int:
        return len(self.primitives)

    def value(
        self,
        point: FloatArray,
    ) -> float:
        """
        Evaluate the contracted Gaussian:

            phi(r) =
                N_c sum_p d_p N_p g_p(r)
        """
        primitive_sum = sum(
            primitive.value(point)
            for primitive in self.primitives
        )

        return float(
            self.contraction_normalization
            * primitive_sum
        )

    def __call__(
        self,
        point: FloatArray,
    ) -> float:
        return self.value(point)
