import json

from common.basis import build_basis_from_dict
from common.molecule import Molecule


with open(
    "h2.json",
    mode="r",
    encoding="utf-8",
) as file:
    data = json.load(file)


molecule = Molecule.from_dict(
    data["molecule"]
)

basis = build_basis_from_dict(
    atoms=molecule.atoms,
    data=data["basis"],
)

print(basis.name)
print(basis.n_functions)