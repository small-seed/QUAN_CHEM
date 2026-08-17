from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


@dataclass(frozen=True)
class Atom:
    """
    Atomic nucleus.

    Parameters
    ----------
    symbol
        Chemical element symbol.

    position
        Nuclear position in bohr, with shape (3,).
    """

    symbol: str
    position: FloatArray

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().capitalize()

        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(
                f"Unsupported chemical element: {symbol}"
            )

        position = np.asarray(
            self.position,
            dtype=np.float64,
        ).copy()

        if position.shape != (3,):
            raise ValueError(
                "Atomic position must have shape (3,)."
            )

        if not np.all(np.isfinite(position)):
            raise ValueError(
                "Atomic position must contain finite coordinates."
            )

        position.setflags(write=False)

        object.__setattr__(
            self,
            "symbol",
            symbol,
        )

        object.__setattr__(
            self,
            "position",
            position,
        )

    @property
    def atomic_number(self) -> int:
        """
        Nuclear charge Z.
        """
        return ATOMIC_NUMBERS[self.symbol]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> Atom:
        """
        Create an Atom from JSON-compatible data.

        Expected structure:

            {
                "symbol": "H",
                "position": [0.0, 0.0, 0.7]
            }
        """
        try:
            symbol = data["symbol"]
            position = data["position"]
        except KeyError as error:
            raise ValueError(
                f"Missing atom field: {error.args[0]}"
            ) from error

        return cls(
            symbol=str(symbol),
            position=np.asarray(
                position,
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True)
class Molecule:
    """
    Molecular system with fixed nuclei.

    Parameters
    ----------
    atoms
        Atomic nuclei forming the molecule.

    charge
        Total molecular charge.

    multiplicity
        Spin multiplicity 2S + 1.
    """

    atoms: tuple[Atom, ...]
    charge: int = 0 "use to compute the number of electron in ion"
    multiplicity: int = 1

    def __post_init__(self) -> None:
        atoms = tuple(self.atoms)

        if not atoms:
            raise ValueError(
                "A molecule must contain at least one atom."
            )

        if not all(
            isinstance(atom, Atom)
            for atom in atoms
        ):
            raise TypeError(
                "All molecule entries must be Atom objects."
            )

        if not isinstance(self.charge, int):
            raise TypeError(
                "Molecular charge must be an integer."
            )

        if (
            not isinstance(self.multiplicity, int)
            or self.multiplicity < 1
        ):
            raise ValueError(
                "Spin multiplicity must be a positive integer."
            )

        object.__setattr__(
            self,
            "atoms",
            atoms,
        )

        if self.n_electrons <= 0:
            raise ValueError(
                "The molecule must contain at least one electron."
            )

        spin_difference = self.multiplicity - 1

        if spin_difference > self.n_electrons:
            raise ValueError(
                "Multiplicity is incompatible "
                "with the electron count."
            )

        if (
            self.n_electrons + spin_difference
        ) % 2 != 0:
            raise ValueError(
                "Charge and multiplicity are incompatible."
            )

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def nuclear_charge(self) -> int:
        """
        Total nuclear charge:

            sum_A Z_A
        """
        return sum(
            atom.atomic_number
            for atom in self.atoms
        )

    @property
    def n_electrons(self) -> int:
        """
        Total number of electrons:

            N_e = sum_A Z_A - molecular charge
        """
        return self.nuclear_charge - self.charge

    @property
    def n_alpha(self) -> int:
        """
        Number of alpha-spin electrons.
        """
        return (
            self.n_electrons
            + self.multiplicity
            - 1
        ) // 2

    @property
    def n_beta(self) -> int:
        """
        Number of beta-spin electrons.
        """
        return self.n_electrons - self.n_alpha

    def nuclear_repulsion_energy(self) -> float:
        """
        Nuclear repulsion energy in atomic units:

            E_NN = sum_{A<B} Z_A Z_B / R_AB
        """
        energy = 0.0

        for index_a, atom_a in enumerate(self.atoms):
            for atom_b in self.atoms[index_a + 1:]:
                displacement = (
                    atom_a.position
                    - atom_b.position
                )

                distance = float(
                    np.linalg.norm(displacement)
                )

                if distance <= 0.0:
                    raise ValueError(
                        "Two nuclei cannot occupy "
                        "the same position."
                    )

                energy += (
                    atom_a.atomic_number
                    * atom_b.atomic_number
                    / distance
                )

        return float(energy)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> Molecule:
        """
        Create a Molecule from JSON-compatible data.

        Expected structure:

            {
                "charge": 0,
                "multiplicity": 1,
                "atoms": [
                    {
                        "symbol": "H",
                        "position": [0.0, 0.0, -0.7]
                    },
                    {
                        "symbol": "H",
                        "position": [0.0, 0.0, 0.7]
                    }
                ]
            }
        """
        try:
            atom_data = data["atoms"]
        except KeyError as error:
            raise ValueError(
                "Missing molecule field: atoms"
            ) from error

        atoms = tuple(
            Atom.from_dict(atom)
            for atom in atom_data
        )

        return cls(
            atoms=atoms,
            charge=int(
                data.get("charge", 0)
            ),
            multiplicity=int(
                data.get("multiplicity", 1)
            ),
        )