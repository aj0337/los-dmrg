import numpy as np
from ase.io import write
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import subdiagonalize_atoms, lowdin_rotation, rotate_matrix
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

SUBDIAG_SPECIES = ("C", "H")
subdiag_indices = get_species_indices(atoms, SUBDIAG_SPECIES)
Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)

# Extract the basis for the subdiagonalized region and get their indices
basis_subdiag_region = basis[subdiag_indices]
index_subdiag_region = basis_subdiag_region.get_indices()

# Define the active region within the subdiagonalized species (C 2 pz in this case)
active = {'C':[3]}
extract_active_region = basis_subdiag_region.extract().take(active)
index_active_region = index_subdiag_region[extract_active_region]

Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)

for idx in index_active_region:
    if Usub[idx-1,idx] < 0.: # change sign
        Usub[:,idx] *= -1

# Rotate matrices
H_subdiagonal = rotate_matrix(H_lcao, Usub)
S_subdiagonal = rotate_matrix(S_lcao, Usub)

lowdin = True
if lowdin:
    Ulow = lowdin_rotation(H_subdiagonal, S_subdiagonal, index_active_region)
    U = Usub.dot(Ulow)

orbital_map = [
    {
        'C': 3
    },
]

folder_path = 'los_lowdin_cube'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)


for orbital in orbital_map:
    orbital_idx = basis.extract().take(orbital)
    los = LOs(Ulow.T, lcao)

    for key, value in orbital.items():
        for w_G in los.get_orbitals(orbital_idx):
            write(f"{folder_path}/lo1_{key}_{value}.cube", atoms, data=w_G)

for orbital in orbital_map:
    orbital_idx = basis.extract().take(orbital)
    los = LOs(U.T, lcao)

    for key, value in orbital.items():
        for w_G in los.get_orbitals(orbital_idx):
            write(f"{folder_path}/lo2_{key}_{value}.cube", atoms, data=w_G)
