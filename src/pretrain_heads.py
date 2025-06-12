import torch
import torch.nn as nn
import random
from torch_scatter import scatter_add
from fragnet.model.gat.gat2 import FragNet


class PretrainTask(nn.Module):
    """
    This function was copied from
    https://github.com/LARS-research/3D-PGT/blob/main/graphgps/head/pretrain_task.py
    and modified

    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in=128, dim_out=1, L=2, netType=1):
        super().__init__()
        self.netType=netType

        # bond_length
        self.bl_reduce_layer = nn.Linear(dim_in * 3, dim_in)
        list_bl_layers = [
            nn.Linear(dim_in // 2**l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_bl_layers.append(nn.Linear(dim_in // 2**L, dim_out, bias=True))
        self.bl_layers = nn.ModuleList(list_bl_layers)

        # bond_angle
        list_ba_layers = [
            nn.Linear(dim_in // 2**l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_ba_layers.append(nn.Linear(dim_in // 2**L, dim_out, bias=True))
        self.ba_layers = nn.ModuleList(list_ba_layers)

        # dihedral_angle
        list_da_layers = [
            nn.Linear(dim_in // 2**l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_da_layers.append(nn.Linear(dim_in // 2**L, dim_out, bias=True))
        self.da_layers = nn.ModuleList(list_da_layers)

        # graph-level prediction (energy)
        list_FC_layers = [
            nn.Linear(dim_in * 2 // 2**l, dim_in * 2 // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(dim_in * 2 // 2**L, dim_out, bias=True))
        
        if self.netType == 1: # atom-level + frag-level
            list_FC_layers.append(nn.Linear(dim_in * 2 // 2**L, 64, bias=True)) #恢复64

        if self.netType == 2 or self.netType == 3: # atom-level or frag-level
            list_FC_layers.append(nn.Linear(dim_in, 64, bias=True)) #仅仅通过atom表示或者frag表示来恢复64

        self.FC_layers = nn.ModuleList(list_FC_layers)

        self.L = L
        self.activation = nn.ReLU()

    def _apply_index(self, batch):
        return batch.bond_length, batch.distance

    def forward(self, x_atoms, x_frags, edge_attr, batch):
        edge_index = batch["edge_index"]

        bond_length_pred = torch.concat(
            (x_atoms[edge_index.T][:, 0, :], x_atoms[edge_index.T][:, 1, :], edge_attr),
            axis=1,
        )
        bond_length_pred = self.bl_reduce_layer(bond_length_pred)
        for l in range(self.L + 1):
            bond_length_pred = self.activation(bond_length_pred)
            bond_length_pred = self.bl_layers[l](bond_length_pred)

        # bond_angle
        bond_angle_pred = x_atoms
        for l in range(self.L):
            bond_angle_pred = self.ba_layers[l](bond_angle_pred)
            bond_angle_pred = self.activation(bond_angle_pred)
        bond_angle_pred = self.ba_layers[self.L](bond_angle_pred)

        # dihedral_angle
        dihedral_angle_pred = edge_attr
        for l in range(self.L):
            dihedral_angle_pred = self.da_layers[l](dihedral_angle_pred)
            dihedral_angle_pred = self.activation(dihedral_angle_pred)
        dihedral_angle_pred = self.da_layers[self.L](dihedral_angle_pred)

        # total energy
        # graph_rep = self.pooling_fun(batch.x, batch.batch)

        x_frags_pooled = scatter_add(src=x_frags, index=batch["frag_batch"], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch["batch"], dim=0)

        graph_rep = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        for l in range(self.L):
            graph_rep = self.FC_layers[l](graph_rep)
            graph_rep = self.activation(graph_rep)

        graph_rep1 = self.FC_layers[self.L](graph_rep)

        if self.netType == 1: # atom-level + frag-level
            graph_rep2 = self.FC_layers[self.L+1](graph_rep)
        if self.netType == 2: # atom-leval only
            graph_rep2 = self.FC_layers[self.L+1](x_atoms_pooled)
        if self.netType == 3: # frag-level only
            graph_rep2 = self.FC_layers[self.L+1](x_frags_pooled)

        return bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep1, graph_rep2


class FragNetPreTrain(nn.Module):

    def __init__(
        self,
        num_layer=4,
        drop_ratio=0.15,
        num_heads=4,
        emb_dim=128,
        atom_features=167,
        frag_features=167,
        edge_features=16,
        fedge_in=6,
        fbond_edge_in=6,
        nt=1
    ):
        super(FragNetPreTrain, self).__init__()

        self.pretrain = FragNet(
            num_layer=num_layer,
            drop_ratio=drop_ratio,
            num_heads=num_heads,
            emb_dim=emb_dim,
            atom_features=atom_features,
            frag_features=frag_features,
            edge_features=edge_features,
            fedge_in=fedge_in,
            fbond_edge_in=fbond_edge_in,
        )
        self.head = PretrainTask(dim_in=128, dim_out=1, netType=nt) #  def __init__(self, dim_in=128, dim_out=1, L=2, netType=1):

    def forward(self, batch):

        x_atoms, x_frags, e_edge, e_fedge = self.pretrain(batch)
        bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep1, graph_rep2 = self.head(
            x_atoms, x_frags, e_edge, batch
        )

        return bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep1, graph_rep2


