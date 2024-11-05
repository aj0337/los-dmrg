import numpy as np
from ase.io import write
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import subdiagonalize_atoms
from qttools.gpaw.los import LOs
import os


data_folder = "output"

def get_species_indices(atoms,species):
    indices = []
    for element in species:
        element_indices = atoms.symbols.search(element)
        indices.extend(element_indices)
    return sorted(indices)

gpwfile = f'{data_folder}/struct.gpw'

atoms, calc = restart(gpwfile, txt=None)
lcao = LCAOwrap(calc)

E_fermi = calc.get_fermi_level()
H_lcao = lcao.get_hamiltonian()
S_lcao = lcao.get_overlap()
H_lcao -= E_fermi * S_lcao

nao_a = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_a)

orbital_map = [
    {
        'C': 3
    },
]

folder_path = f'{data_folder}/los_cube'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

SUBDIAG_SPECIES = ("C", "H")
subdiag_indices = get_species_indices(atoms, SUBDIAG_SPECIES)
Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)

basis_subdiag_region = basis[subdiag_indices]
index_subdiag_region = basis_subdiag_region.get_indices()

for orbital in orbital_map:
    orbital_idx = basis_subdiag_region.extract().take(orbital)
    los = LOs(Usub[:,index_subdiag_region].T, lcao)

    for key, value in orbital.items():
        for w, w_G in enumerate(los.get_orbitals(orbital_idx)):
            write(f"{folder_path}/lo{key}{w}_orb{value}.cube", atoms, data=w_G)
