import rdkit
import rdkit.Chem as Chem

from .chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo
from .vocab import *
from .mol_tree import MolTreeNode

class PropMolTree(object):

    def __init__(self, smiles, props): # add props as arg
        self.smiles = smiles
        self.props = props
        self.mol = get_mol(smiles)

        cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        root = 0
        for i,c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0: root = i

        for x,y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

        for i,node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1: #Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()

def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx: continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1
