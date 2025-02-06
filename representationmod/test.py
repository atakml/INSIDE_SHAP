import re
import csv
from torch_geometric.data import Data
import torch
from rdkit import Chem
from torch_geometric.utils import from_networkx
import networkx as nx

def smiles_to_pyg(smiles):
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), x=torch.tensor([atom.GetAtomicNum()], dtype=torch.float))

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    data = from_networkx(G)
    return data

def filter_molecules(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            smiles = row[1]  # Assuming the second column contains the SMILES strings
            pyg_data = smiles_to_pyg(smiles)
            if pyg_data is None:
                continue

            atom_types = pyg_data.x.squeeze().tolist()
            if not all(atom in [1, 6, 7, 8, 17] for atom in atom_types):
                continue

            num_oxygens = atom_types.count(8)
            num_nitrogens = atom_types.count(7)
            num_chlorines = atom_types.count(17)

            if num_oxygens == 3 and num_nitrogens == 2 and num_chlorines == 1:
                edge_index = pyg_data.edge_index
                nitrogen_pairs = [(u.item(), v.item()) for u, v in zip(edge_index[0], edge_index[1]) if pyg_data.x[u].item() == 7 and pyg_data.x[v].item() == 7]
                oxygen_neighbors = [u.item() for u, v in zip(edge_index[0], edge_index[1]) if pyg_data.x[u].item() == 8 and pyg_data.x[v].item() == 6]

                if len(nitrogen_pairs) > 0 and len(oxygen_neighbors) >= 2:
                    print(smiles)

filter_molecules('representationmod/benzene_smiles.csv')

