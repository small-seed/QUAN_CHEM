from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterator, Mapping

from .gaussian import (
    AngularMomentum,
    ContractedGaussian,
)
from .molecule import Atom


@dataclass(frozen=True)
class ShellDefinition:
    """
    Definition of one atomic basis shell.

    Parameters
    ----------
    angular_momentum
        Total angular momentum L:

            0 -> s
            1 -> p
            2 -> d
            3 -> f

    exponents
        Primitive Gaussian exponents alpha_p.

    coefficients
        Contraction coefficients d_p.
    """

    angular_momentum: int
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        exponents = tuple(
            float(value)
            for value in self.exponents
        )

        coefficients = tuple(
            float(value)
            for value in self.coefficients
        )

        if (
            not isinstance(self.angular_momentum, int)
            or self.angular_momentum < 0
        ):
            raise ValueError(
                "Angular momentum must be "
                "a non-negative integer."
            )

        if not exponents:
            raise ValueError(
                "A shell must contain at least one exponent."
            )

        if len(exponents) != len(coefficients):
            raise ValueError(
                "Exponents and coefficients "
                "must have equal lengths."
            )

        if any(
            not isfinite(exponent)
            or exponent <= 0.0
            for exponent in exponents
        ):
            raise ValueError(
                "All Gaussian exponents must be "
                "finite positive numbers."
            )

        if any(
            not isfinite(coefficient)
            for coefficient in coefficients
        ):
            raise ValueError(
                "All contraction coefficients "
                "must be finite numbers."
            )

        object.__setattr__(
            self,
            "exponents",
            exponents,
        )

        object.__setattr__(
            self,
            "coefficients",
            coefficients,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> ShellDefinition:
        """
        Create one shell definition from JSON-compatible data.

        Expected structure:

            {
                "angular_momentum": 0,
                "exponents": [...],
                "coefficients": [...]
            }
        """
        try:
            angular_momentum = data["angular_momentum"]
            exponents = data["exponents"]
            coefficients = data["coefficients"]

        except KeyError as error:
            raise ValueError(
                f"Missing shell field: {error.args[0]}"
            ) from error

        return cls(
            angular_momentum=int(
                angular_momentum
            ),
            exponents=tuple(exponents),
            coefficients=tuple(coefficients),
        )


@dataclass(frozen=True)
class BasisSet:
    """
    AO basis set of an entire molecule.

    Each function is one contracted Cartesian Gaussian.
    """

    name: str
    functions: tuple[ContractedGaussian, ...]

    def __post_init__(self) -> None:
        name = self.name.strip()

        functions = tuple(
            self.functions
        )

        if not name:
            raise ValueError(
                "Basis-set name cannot be empty."
            )

        if not functions:
            raise ValueError(
                "A basis set must contain "
                "at least one basis function."
            )

        if not all(
            isinstance(function, ContractedGaussian)
            for function in functions
        ):
            raise TypeError(
                "All basis functions must be "
                "ContractedGaussian objects."
            )

        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "functions",
            functions,
        )

    def __len__(self) -> int:
        return len(self.functions)

    def __getitem__(
        self,
        index: int,
    ) -> ContractedGaussian:
        return self.functions[index]

    def __iter__(
        self,
    ) -> Iterator[ContractedGaussian]:
        return iter(self.functions)


def cartesian_components(
    total_angular_momentum: int,
) -> tuple[AngularMomentum, ...]:
    """
    Generate all Cartesian angular-momentum components
    satisfying:

        l + m + n = L

    Examples
    --------
    L = 0:

        ((0, 0, 0),)

    L = 1:

        (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        )
    """
    if (
        not isinstance(total_angular_momentum, int)
        or total_angular_momentum < 0
    ):
        raise ValueError(
            "Total angular momentum must be "
            "a non-negative integer."
        )

    components: list[AngularMomentum] = []

    for l in range(
        total_angular_momentum,
        -1,
        -1,
    ):
        remaining = (
            total_angular_momentum - l
        )

        for m in range(
            remaining,
            -1,
            -1,
        ):
            n = remaining - m

            components.append(
                (l, m, n)
            )

    return tuple(components)


def build_atom_basis(
    atom: Atom,
    shells: tuple[ShellDefinition, ...],
    atom_index: int,
) -> tuple[ContractedGaussian, ...]:
    """
    Build all AO basis functions located on one atom.
    """
    functions: list[ContractedGaussian] = []

    for shell_index, shell in enumerate(
        shells,
        start=1,
    ):
        components = cartesian_components(
            shell.angular_momentum
        )

        for component in components:
            label = (
                f"{atom.symbol}{atom_index} "
                f"shell={shell_index} "
                f"angular={component}"
            )

            basis_function = (
                ContractedGaussian.from_parameters(
                    exponents=shell.exponents,
                    coefficients=shell.coefficients,
                    center=atom.position,
                    angular_momentum=component,
                    label=label,
                )
            )

            functions.append(
                basis_function
            )

    return tuple(functions)


def build_basis_from_dict(
    atoms: tuple[Atom, ...],
    data: Mapping[str, Any],
) -> BasisSet:
    """
    Build the molecular AO basis from JSON-compatible data.

    Expected structure:

        {
            "name": "sto-3g",
            "elements": {
                "H": [
                    {
                        "angular_momentum": 0,
                        "exponents": [...],
                        "coefficients": [...]
                    }
                ]
            }
        }
    """
    try:
        basis_name = data["name"]
        element_data = data["elements"]

    except KeyError as error:
        raise ValueError(
            f"Missing basis field: {error.args[0]}"
        ) from error

    if not isinstance(element_data, Mapping):
        raise ValueError(
            "'elements' must be a mapping "
            "from element symbols to shell lists."
        )

    shell_library: dict[
        str,
        tuple[ShellDefinition, ...],
    ] = {}

    for symbol, shell_data_list in (
        element_data.items()
    ):
        if not isinstance(
            shell_data_list,
            list,
        ):
            raise ValueError(
                f"Basis data for {symbol} "
                "must be a list of shells."
            )

        shell_library[str(symbol)] = tuple(
            ShellDefinition.from_dict(
                shell_data
            )
            for shell_data in shell_data_list
        )

    functions: list[ContractedGaussian] = []

    for atom_index, atom in enumerate(
        atoms,
        start=1,
    ):
        if atom.symbol not in shell_library:
            raise ValueError(
                f"No basis data found for "
                f"element {atom.symbol}."
            )

        atom_functions = build_atom_basis(
            atom=atom,
            shells=shell_library[atom.symbol],
            atom_index=atom_index,
        )

        functions.extend(
            atom_functions
        )

    return BasisSet(
        name=str(basis_name),
        functions=tuple(functions),
    )