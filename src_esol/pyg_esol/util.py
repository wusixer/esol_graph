#!/usr/bin/env python
"""util for esol data manipulation"""


import rdkit
from typing import Tuple
import torch
import torch_geometric

# dictionary to numerically encode the categorical variables
atom_feature_map = {
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
}

bond_feature_map = {
    "bond_type": [
        "misc",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}


def get_one_atom_feature(atom: rdkit.Chem.rdchem.Atom) -> list:
    """Get features from one atom

    Args:
        atom (rdkit.Chem.rdchem.Atom): one atom returned by mol.GetAtoms()

    Returns:
        list: nine features include atom number, chirality, total degree. etc
    """

    return [
        atom.GetAtomicNum(),
        atom_feature_map["chirality"].index(str(atom.GetChiralTag())),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetNumRadicalElectrons(),
        atom_feature_map["hybridization"].index(str(atom.GetHybridization())),
        [1 if atom.GetIsAromatic() else 0][0],
        [1 if atom.IsInRing() else 0][0],
    ]


def get_one_bond_feature(bond: rdkit.Chem.rdchem.Bond) -> Tuple[list, list]:
    """get features from one bond

    Args:
        bond (rdkit.Chem.rdchem.Bond): one bond returned by mol.GetBonds()

    Returns:
        Tuple[list,list]:
        edge_indices: a Tuple denotes the direction of the connection
                    [[source_node, des_node], [des_node, source_node]]
        edge_attrs: a Tuple shows the bond attribute of inward edge and outward edge
    """
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    edge_indices = [[i, j], [j, i]]
    bond_feature = [
        bond_feature_map["bond_type"].index(str(bond.GetBondType())),
        bond_feature_map["stereo"].index(str(bond.GetStereo())),
        [1 if bond.GetIsConjugated() else 0][0],
    ]
    edge_attrs = [bond_feature, bond_feature]

    return edge_indices, edge_attrs


def convert_one_smile_to_graph(
    one_smile: str, y: float
) -> torch_geometric.data.data.Data:
    """Convert a smile string with its associated target value into a torch_geometric.homogeneous graph

    Args:
        one_smile (str): a str reflect the molecule, such as 'Cc1occc1C(=O)Nc2ccccc2'
        y (float): the target value of associated one_smile

    Returns:
        torch_geometric.data.data.Data: a torch_geometric homogeneous graph
    """

    y = torch.tensor(y, dtype=torch.float).view(1, -1)

    mol = Chem.MolFromSmiles(one_smile)

    # get atom features: atomic_num, chirality, degreee, formal_charge,
    # num_hs, num_radical_electrons, hybridization, is_aromatic, is_in_ring
    atom_features = list(map(get_one_atom_feature, mol.GetAtoms()))
    x = torch.tensor(atom_features, dtype=torch.long).view(-1, 9)

    # get edge features
    edge_indices = [
        i[0] for i in [get_one_bond_feature(edge)[0] for edge in mol.GetBonds()]
    ]
    edge_attrs = [
        i[0] for i in [get_one_bond_feature(edge)[1] for edge in mol.GetBonds()]
    ]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)
    # print(f"edge_index before sort is {edge_index}")

    # Sort indices, so that the edge_index[0] (source nodes) are ascending order
    # to reduce jump in memory
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
        # print(f"edge_index after sort is {edge_index}")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smile=one_smile)
    return data
