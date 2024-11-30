from __future__ import print_function
import os
from ase import *
from ase.io import read
from gpaw import *

# Load atomic structure
atoms = read('struct.xyz')

# Define basis set for atoms
basis = {'H': 'dzp', 'C': 'dzp'}

U = 11
hubbard = {'C': ':p,11.0'}

# Define the output directory
output_dir = f'output/gpaw/HubbardU{U}'
os.makedirs(output_dir, exist_ok=True)

# Set up GPAW calculator
calc = GPAW(h=0.2,
            xc='PBE',
            basis=basis,
            occupations=FermiDirac(width=0.2),
            kpts=(1, 1, 1),
            mode='lcao',
            txt=os.path.join(output_dir, 'struct.txt'),  # Output path for GPAW text output
            mixer=Mixer(0.1, 5, weight=100.0),
            symmetry={'point_group': False, 'time_reversal': True},
            setups=hubbard)

atoms.set_calculator(calc)

# Run calculation
atoms.get_potential_energy()

# Write the calculator state to a .gpw file in the output directory
calc.write(os.path.join(output_dir, 'struct.gpw'), mode='all')

# Get and save the Fermi level to a text file in the output directory
fermi = calc.get_fermi_level()
with open(os.path.join(output_dir, 'fermi_struct.txt'), 'w') as f:
    print(repr(fermi), file=f)
