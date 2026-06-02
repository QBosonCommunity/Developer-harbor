# -*- coding: utf-8 -*-
"""
Project: Quantum-Classical Hybrid Framework - Molecular Visualization
Version: 2.0 (High-Resolution Journal Edition)

Description:
    This independent rendering module translates abstract graph-theoretical results
    into publication-quality molecular visualizations. It performs two distinct tasks:
    1. Graph Topology Rendering: Maps the discrete adjacency matrix into an optimized
       2D force-directed node-link diagram (NetworkX).
    2. Physical Conformation Rendering: Projects the 3D MMFF94-optimized atomic
       coordinates onto a high-resolution 2D chemical plane (RDKit).

Outputs:
    - molecule_structure_2d_topology.png
    - molecule_structure_3d_conformer.png

Author: Zhong Xuan (Hunan Agricultural University / National University of Defense Technology)
"""

import logging
from typing import List, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Chemical Informatics Engine
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')  # Suppress C++ backend warnings
except ImportError:
    raise ImportError("Critical Dependency Error: 'rdkit' is required for physical conformer rendering.")

# =========================================================================
# System Configuration & Logging Setup
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("MolecularRenderer")

# Hardware Optimum Candidate Parameters (Extracted from CIM Logs)
FINAL_ENERGY: float = -78.5490
MOLECULAR_FORMULA: List[str] = ['C', 'C', 'C', 'C', 'C', 'C', 'O', 'O']

# Graph Topology Edges (Bicyclic backbone with functional side chains)
STRUCTURAL_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # Main ring structure
    (0, 6), (3, 7),  # Functional side chains
    (1, 4)  # Internal cross-link for structural rigidity
]


# =========================================================================
# Visualization Modules
# =========================================================================
def render_graph_topology(atoms: List[str], edges: List[Tuple[int, int]], energy_score: float) -> None:
    """
    Renders an abstract mathematical graph topology using force-directed layout algorithms.
    Applies strict academic aesthetics with high-contrast elemental node coloring.
    """
    logger.info("Initializing NetworkX mathematical graph renderer...")

    graph = nx.Graph()
    color_palette = []
    node_labels = {}

    # Construct Nodes and Map Chemical Aesthetics
    for i, element in enumerate(atoms):
        graph.add_node(i)
        node_labels[i] = element
        if element == 'C':
            color_palette.append('#5D6D7E')  # Muted Carbon Grey
        elif element == 'O':
            color_palette.append('#E74C3C')  # Academic Alizarin Red
        else:
            color_palette.append('#95A5A6')

    graph.add_edges_from(edges)

    # Figure Configuration
    plt.figure(figsize=(8, 6), facecolor='white')

    # Spring layout with adjusted repulsive force (k) for better spacing
    layout_positions = nx.spring_layout(graph, seed=42, k=1.2)

    # Render Components
    nx.draw_networkx_nodes(
        graph, layout_positions, node_color=color_palette,
        node_size=1200, edgecolors='black', linewidths=1.5
    )
    nx.draw_networkx_edges(
        graph, layout_positions, width=3.5, edge_color='#7F8C8D', alpha=0.8
    )
    nx.draw_networkx_labels(
        graph, layout_positions, node_labels,
        font_size=16, font_color='white', font_weight='bold'
    )

    plt.title(f"Quantum-Optimized Molecular Topology\nCalculated Free Energy: {energy_score:.4f} kcal/mol",
              fontsize=16, fontweight='bold', pad=15)
    plt.axis('off')

    output_filename = "molecule_structure_2d_topology.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    logger.info(f"Topological graph successfully rendered and saved to '{output_filename}'")


def render_chemical_conformer(atoms: List[str], edges: List[Tuple[int, int]], energy_score: float) -> None:
    """
    Constructs a valid RDKit molecule object, projects it into 3D Euclidean space,
    optimizes it using MMFF94 physics, and renders a high-resolution 2D projection.
    """
    logger.info("Initializing RDKit physics engine and conformer rendering pipeline...")

    editable_mol = Chem.RWMol()
    atom_indices = []

    for element in atoms:
        atom_indices.append(editable_mol.AddAtom(Chem.Atom(element)))

    for u, v in edges:
        editable_mol.AddBond(atom_indices[u], atom_indices[v], Chem.BondType.SINGLE)

    static_mol = editable_mol.GetMol()
    Chem.SanitizeMol(static_mol)

    # 3D Conformer Generation (ETKDG) and Force Field Relaxation
    hydrogenated_mol = Chem.AddHs(static_mol)
    AllChem.EmbedMolecule(hydrogenated_mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(hydrogenated_mol)

    # High-Resolution RDKit Rendering Engine Configuration
    img_resolution = (800, 800)
    drawer = rdMolDraw2D.MolDraw2DCairo(img_resolution[0], img_resolution[1])

    # Apply Academic Drawing Options
    draw_opts = drawer.drawOptions()
    draw_opts.bondLineWidth = 3
    draw_opts.minFontSize = 18
    draw_opts.clearBackground = True
    draw_opts.legendFontSize = 24

    drawer.DrawMolecule(
        hydrogenated_mol,
        legend=f"MMFF94 3D Conformer Projection (Energy: {energy_score:.4f})"
    )
    drawer.FinishDrawing()

    output_filename = "molecule_structure_3d_conformer.png"
    with open(output_filename, 'wb') as f:
        f.write(drawer.GetDrawingText())

    logger.info(f"Chemical conformer projection successfully rendered and saved to '{output_filename}'")


# =========================================================================
# Main Execution Pipeline
# =========================================================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("STARTING MOLECULAR VISUALIZATION PIPELINE")
    logger.info("=" * 60)

    try:
        render_graph_topology(MOLECULAR_FORMULA, STRUCTURAL_EDGES, FINAL_ENERGY)
        render_chemical_conformer(MOLECULAR_FORMULA, STRUCTURAL_EDGES, FINAL_ENERGY)
        logger.info("=" * 60)
        logger.info("SUCCESS: All high-resolution figures have been generated.")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Visualization Pipeline Terminated Unexpectedly: {e}")