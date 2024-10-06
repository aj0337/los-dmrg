import numpy as np
from ase.io import write
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import subdiagonalize_atoms
from qttools.gpaw.los import LOs
import os


def get_species_indices(atoms,species):
    indices = []
    for element in species:
        element_indices = atoms.symbols.search(element)
        indices.extend(element_indices)
    return sorted(indices)

gpwfile = './struct.gpw'

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
        'C': 0
    },
    {
        'C': 1
    },
    {
        'C': 2
    },
    {
        'C': 3
    },
    {
        'C': 4
    },
    {
        'C': 5
    },
    {
        'C': 6
    },
    {
        'C': 7
    },
    {
        'C': 8
    },
    {
        'C': 9
    },
    {
        'C': 10
    },
    {
        'C': 11
    },
    {
        'C': 12
    },
    {
        'H': 0
    },
    {
        'H': 1
    },
    {
        'H': 2
    },
    {
        'H': 3
    },
    {
        'H': 4
    },
]

folder_path = 'los_cube_files'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

SUBDIAG_SPECIES = ("C", "H")
subdiag_indices = get_species_indices(atoms, SUBDIAG_SPECIES)
Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)

for orbital in orbital_map:
    orbital_idx = basis.extract().take(orbital)
    los = LOs(Usub.T, lcao)

    for key, value in orbital.items():
        for w_G in los.get_orbitals(orbital_idx):
            write(f"{folder_path}/lo_{key}_{value}.cube", atoms, data=w_G)
