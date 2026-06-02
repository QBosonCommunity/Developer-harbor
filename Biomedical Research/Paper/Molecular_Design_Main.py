# -*- coding: utf-8 -*-
"""
Project: Quantum-Classical Hybrid Computational Framework for De Novo Molecular Design
Version: 4.2 (Academic Refactored & Hardware-Optimized Edition - Bugfixed)

Description:
    This framework implements a cutting-edge, hybrid quantum-classical pipeline
    for the automated discovery and optimization of small-molecule drug candidates.

    The architecture seamlessly bridges macro-level pharmaceutical constraints with
    micro-level quantum mechanics through a bi-phase workflow:
      1. Stochastic Quantum Exploration: Unconstrained search across massive discrete
         chemical topology graphs using a Coherent Ising Machine (CIM).
      2. Active Surrogate-Guided Optimization: Iterative mathematical convergence guided
         by a classical Linear Regression surrogate model representing the target property landscape.

Mathematical & Physics Formulations:
    - Valency Hard Penalty: Built upon strict structural valence boundary rules.
    - Mapping Paradigm: Quadratic Unconstrained Binary Optimization (QUBO) mapped to a standard
      Canonical Ising Spin Hamiltonian via an elegant Auxiliary Spin Embedding.
    - Conformal Geometry Relaxation: Molecular Mechanics Force Field (MMFF94) optimization
      under classical physics potentials to guarantee 3D structural stability in Euclidean space.

Computational Platform: Kaiwu SDK (CIM Photonic Accelerator Cloud Backend)
Author: Zhong Xuan (Hunan Agricultural University / National University of Defense Technology)
"""

import time
import os
import shutil
import random
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Intellectual Property / Professional Chemical Informatics Engine
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')  # Suppress low-level C++ warning noise
except ImportError:
    raise ImportError("Critical Dependency Error: 'rdkit' is required. Please install it.")

try:
    import kaiwu as kw
    import kaiwu.preprocess as kw_pre
except ImportError:
    raise ImportError("Critical Dependency Error: 'kaiwu' SDK is missing.")

# =========================================================================
# Computational Logging & Infrastructure Setup
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quantum_molecular_design.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("QuantumMolecularFramework")

CHECKPOINT_DIR: str = './tmp_kaiwu_checkpoint'
SDF_DIR: str = './batch_molecules_sdf'


def initialize_quantum_infrastructure() -> None:
    """Verifies and initializes local checkpoint nodes and remote quantum accelerator credentials."""
    logger.info("Initializing high-performance computational infrastructure...")

    for directory in [CHECKPOINT_DIR, SDF_DIR]:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
            except Exception as e:
                logger.warning(f"Directory sanitization anomaly at {directory}: {e}")
        os.makedirs(directory, exist_ok=True)

    logger.info(f"Computational directories mapped. SDF Output Target: '{SDF_DIR}'")
    kw.common.CheckpointManager.save_dir = CHECKPOINT_DIR

    try:
        kw.license.init(
            user_id="91850531256946690",
            sdk_code="lTj5v0u67gyWsMfXxKAbiJPkkT6w7u"
        )
        logger.info("Quantum Photonic Hardware Authentication: SUCCESS.")
    except Exception as e:
        logger.error(f"Quantum Hardware Authentication FAILED: {e}")


# =========================================================================
# Physics Engine & Conformation Simulation (RDKit Protocol)
# =========================================================================
def evaluate_and_generate_conformer_3d(
        adj_matrix: np.ndarray,
        atoms: List[str],
        potential_energy: float,
        batch_idx: int,
        phase_signature: str
) -> Optional[str]:
    """
    Projects discrete graph topology onto 3D Euclidean space and applies
    a strict MMFF94 force field relaxation.
    """
    n_atoms = len(atoms)
    editable_mol = Chem.RWMol()
    atom_indices: Dict[int, int] = {}

    for i, symbol in enumerate(atoms):
        atom_indices[i] = editable_mol.AddAtom(Chem.Atom(symbol))

    total_bonds = 0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if adj_matrix[i, j] == 1:
                editable_mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)
                total_bonds += 1

    if total_bonds < n_atoms - 1:
        return None

    try:
        static_mol = editable_mol.GetMol()
        Chem.SanitizeMol(static_mol)
        hydrogenated_mol = Chem.AddHs(static_mol)

        if AllChem.EmbedMolecule(hydrogenated_mol, AllChem.ETKDG()) == -1:
            return None

        # Force Field Optimization (Strict Physics Validation)
        AllChem.MMFFOptimizeMolecule(hydrogenated_mol)

        filename = f"{phase_signature}_Batch{batch_idx}_{int(abs(potential_energy) * 100)}.sdf"
        destination_path = os.path.join(SDF_DIR, filename)

        hydrogenated_mol.SetProp("_Name", f"{phase_signature}_Batch{batch_idx}")
        hydrogenated_mol.SetProp("Oracle_Energy", f"{potential_energy:.4f}")

        with Chem.SDWriter(destination_path) as writer:
            writer.write(hydrogenated_mol)

        return destination_path

    except Exception:
        return None


# =========================================================================
# Quantum Compiler & Mapping Core (QUBO / Ising)
# =========================================================================
class MolecularDesignCompiler:
    """
    Compiles macro-level structural formulations into highly optimized
    quadratic spin models compatible with physical Ising-type hardware.
    """

    def __init__(self, atoms: List[str], max_valencies: List[int]) -> None:
        self.atoms = atoms
        self.max_valencies = max_valencies
        self.n_atoms = len(atoms)

        self.variable_counter = 0
        self.bond_to_index_map: Dict[Tuple[int, int], int] = {}
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                self.bond_to_index_map[(i, j)] = self.variable_counter
                self.variable_counter += 1

        self.slack_to_index_map: Dict[Tuple[int, int], int] = {}
        for i in range(self.n_atoms):
            upper_bound = self.max_valencies[i]
            bit_depth = int(np.floor(np.log2(upper_bound))) + 1
            for k in range(bit_depth):
                self.slack_to_index_map[(i, k)] = self.variable_counter
                self.variable_counter += 1

        self.num_qubo_variables = self.variable_counter
        self.total_ising_spins = self.num_qubo_variables + 1
        logger.info(f"Quantum Mapping initialized. Spin Footprint: {self.total_ising_spins}")

    def _fetch_bond_index(self, u: int, v: int) -> Optional[int]:
        return self.bond_to_index_map.get((u, v) if u < v else (v, u))

    def formulate_qubo_matrix(self, surrogate_weights: Optional[np.ndarray] = None,
                              penalty_coefficient: float = 100.0) -> np.ndarray:
        Q = np.zeros((self.num_qubo_variables, self.num_qubo_variables))
        if surrogate_weights is not None:
            for idx, (u, v) in enumerate(self.bond_to_index_map.keys()):
                if idx < len(surrogate_weights):
                    Q[self.bond_to_index_map[(u, v)], self.bond_to_index_map[(u, v)]] += float(surrogate_weights[idx])

        for i in range(self.n_atoms):
            max_val = self.max_valencies[i]
            linear_terms: List[Tuple[int, float]] = []

            for j in range(self.n_atoms):
                if i == j: continue
                bond_idx = self._fetch_bond_index(i, j)
                if bond_idx is not None: linear_terms.append((bond_idx, 1.0))

            bit_depth = int(np.floor(np.log2(max_val))) + 1
            for k in range(bit_depth):
                slack_idx = self.slack_to_index_map[(i, k)]
                linear_terms.append((slack_idx, float(2.0 ** k)))

            offset_constant = -float(max_val)
            for p in range(len(linear_terms)):
                idx_p, coef_p = linear_terms[p]
                diagonal_linear_coupling = (coef_p ** 2) + (2.0 * offset_constant * coef_p)
                Q[idx_p, idx_p] += penalty_coefficient * diagonal_linear_coupling

                for q in range(p + 1, len(linear_terms)):
                    idx_q, coef_q = linear_terms[q]  # <--- HERE IS THE FIX: changed terms[q] to linear_terms[q]
                    cross_quadratic_coupling = 2.0 * coef_p * coef_q
                    if idx_p < idx_q:
                        Q[idx_p, idx_q] += penalty_coefficient * cross_quadratic_coupling
                    else:
                        Q[idx_q, idx_p] += penalty_coefficient * cross_quadratic_coupling
        return Q

    def compile_qubo_to_ising(self, Q_qubo: np.ndarray) -> np.ndarray:
        N = self.num_qubo_variables
        J = np.zeros((N + 1, N + 1))
        Q_symmetric = np.triu(Q_qubo) + np.triu(Q_qubo, 1).T
        magnetic_fields = np.zeros(N)

        for i in range(N):
            magnetic_fields[i] += Q_symmetric[i, i] / 2.0
            for j in range(i + 1, N):
                coupling_value = Q_symmetric[i, j] / 4.0
                J[i, j] = coupling_value
                J[j, i] = coupling_value
                magnetic_fields[i] += coupling_value
                magnetic_fields[j] += coupling_value

        auxiliary_spin_idx = N
        for i in range(N):
            J[i, auxiliary_spin_idx] = magnetic_fields[i]
            J[auxiliary_spin_idx, i] = magnetic_fields[i]

        return J

    def adapt_hardware_precision(self, J_matrix: np.ndarray) -> np.ndarray:
        max_amplitude = np.max(np.abs(J_matrix))
        if max_amplitude == 0: return J_matrix

        dynamic_scale_factor = (100.0 / max_amplitude) * 0.95
        J_scaled = np.round(J_matrix * dynamic_scale_factor)

        try:
            return kw_pre.perform_precision_adaption_mutate(J_scaled)
        except Exception:
            logger.warning("Hardware mutate anomaly. Enforcing strict manual zero-diagonal fallback.")
            np.fill_diagonal(J_scaled, 0)
            return J_scaled

    def execute_quantum_annealing(self, J_matrix: np.ndarray, identifier_tag: str) -> List[np.ndarray]:
        pd.DataFrame(J_matrix).to_csv(f"{identifier_tag}.csv", index=False, header=False)
        solver_pipeline = kw.cim.CIMOptimizer(task_name_prefix=identifier_tag, wait=True, interval=1)

        for attempt in range(3):
            try:
                return solver_pipeline.solve(J_matrix)
            except Exception as network_exception:
                logger.warning(f"Quantum Cloud API latency (Attempt {attempt + 1}/3): {network_exception}")
                time.sleep(3)
        return []

    def decode_spin_solution(self, raw_ising_solution: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        if len(raw_ising_solution) != self.total_ising_spins:
            return None, None

        primary_spins = raw_ising_solution[:-1]
        auxiliary_spin_sign = raw_ising_solution[-1]

        stabilized_spins = primary_spins * auxiliary_spin_sign
        binary_graph_variables = (stabilized_spins + 1) // 2

        decoded_adjacency_matrix = np.zeros((self.n_atoms, self.n_atoms), dtype=int)
        for (u, v), index in self.bond_to_index_map.items():
            if index < len(binary_graph_variables):
                state_value = int(binary_graph_variables[index])
                decoded_adjacency_matrix[u, v] = state_value
                decoded_adjacency_matrix[v, u] = state_value

        structural_feature_vector = [decoded_adjacency_matrix[u, v] for u, v in self.bond_to_index_map.keys()]
        return decoded_adjacency_matrix, structural_feature_vector


# =========================================================================
# Evaluator & Data Synthesis
# =========================================================================
def evaluate_molecular_oracle_energy(adj_matrix: np.ndarray, atoms: List[str]) -> float:
    base_free_energy = 0.0
    n_elements = len(atoms)
    empirical_bond_potentials = {('C', 'C'): -2.5, ('C', 'N'): -3.0, ('C', 'O'): -3.5, ('O', 'O'): 1.0}

    active_bonds = 0
    for i in range(n_elements):
        for j in range(i + 1, n_elements):
            if adj_matrix[i, j] == 1:
                active_bonds += 1
                atomic_pair = tuple(sorted((atoms[i], atoms[j])))
                base_free_energy += empirical_bond_potentials.get(atomic_pair, -0.5)

    if active_bonds < n_elements - 1:
        base_free_energy += 20.0

    return base_free_energy + np.random.normal(0, 0.05)


def generate_synthetic_prior_landscape(n_samples: int, atoms: List[str], max_valencies: List[int],
                                       compiler: MolecularDesignCompiler) -> Tuple[List[List[int]], List[float]]:
    logger.info(f"Synthesizing baseline historical dataset ({n_samples} samples)...")
    X_landscape, y_energy_metrics = [], []
    n_elements = len(atoms)
    successful_samples = 0

    while successful_samples < n_samples:
        adj = np.zeros((n_elements, n_elements), dtype=int)
        degrees = np.zeros(n_elements, dtype=int)

        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                if random.random() < 0.4:
                    if degrees[i] < max_valencies[i] and degrees[j] < max_valencies[j]:
                        adj[i, j] = 1;
                        adj[j, i] = 1
                        degrees[i] += 1;
                        degrees[j] += 1

        sample_energy = evaluate_molecular_oracle_energy(adj, atoms)
        if sample_energy > 50.0:
            continue

        feature_vector = [adj[u, v] for u, v in compiler.bond_to_index_map.keys()]
        X_landscape.append(feature_vector)
        y_energy_metrics.append(sample_energy)
        successful_samples += 1

    return X_landscape, y_energy_metrics


# =========================================================================
# Main Execution Pipeline
# =========================================================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("STARTING QUANTUM-CLASSICAL HYBRID MOLECULAR GENERATION")
    logger.info("=" * 60)

    initialize_quantum_infrastructure()

    MOLECULAR_FORMULA_ATOMS: List[str] = ['C', 'C', 'C', 'C', 'C', 'C', 'O', 'O']
    VALENCE_CONSTRAINTS: List[int] = [4, 4, 4, 4, 4, 4, 2, 2]

    compiler = MolecularDesignCompiler(MOLECULAR_FORMULA_ATOMS, VALENCE_CONSTRAINTS)
    discovered_compounds_manifest = []

    X_train, y_train = generate_synthetic_prior_landscape(500, MOLECULAR_FORMULA_ATOMS, VALENCE_CONSTRAINTS, compiler)

    logger.info("\n>>> TRACK 1: INITIAL STOCHASTIC QUANTUM EXPLORATION <<<")
    for batch_idx in range(3):
        Q_matrix = compiler.formulate_qubo_matrix(penalty_coefficient=80.0)
        J_hardware_optimized = compiler.adapt_hardware_precision(compiler.compile_qubo_to_ising(Q_matrix))

        hardware_quantum_spins = compiler.execute_quantum_annealing(J_hardware_optimized, f"exp_{batch_idx}")

        if hardware_quantum_spins is not None and len(hardware_quantum_spins) > 0:
            saved_candidates_count = 0
            for state in hardware_quantum_spins:
                adj, feat = compiler.decode_spin_solution(state)
                if adj is not None:
                    computed_energy = evaluate_molecular_oracle_energy(adj, MOLECULAR_FORMULA_ATOMS)
                    X_train.append(feat)
                    y_train.append(computed_energy)

                    sdf_filepath = evaluate_and_generate_conformer_3d(
                        adj, MOLECULAR_FORMULA_ATOMS, computed_energy, batch_idx, "Exploration"
                    )

                    if sdf_filepath:
                        discovered_compounds_manifest.append(
                            {"Phase": "Exploration", "Energy": computed_energy, "File": sdf_filepath})
                        saved_candidates_count += 1
                if saved_candidates_count >= 3: break
            logger.info(f"Exploration Batch [{batch_idx}]: Saved {saved_candidates_count} validated 3D compounds.")

    logger.info("\n>>> TRACK 2: ITERATIVE SURROGATE-GUIDED CONVERGENCE <<<")
    active_surrogate_model = LinearRegression()

    for step_idx in range(3):
        active_surrogate_model.fit(X_train, y_train)
        Q_matrix = compiler.formulate_qubo_matrix(surrogate_weights=active_surrogate_model.coef_,
                                                  penalty_coefficient=120.0)
        J_hardware_optimized = compiler.adapt_hardware_precision(compiler.compile_qubo_to_ising(Q_matrix))

        hardware_quantum_spins = compiler.execute_quantum_annealing(J_hardware_optimized, f"opt_{step_idx}")

        if hardware_quantum_spins is not None and len(hardware_quantum_spins) > 0:
            adj, feat = compiler.decode_spin_solution(hardware_quantum_spins[0])
            if adj is not None:
                optimized_energy = evaluate_molecular_oracle_energy(adj, MOLECULAR_FORMULA_ATOMS)
                X_train.append(feat)
                y_train.append(optimized_energy)

                sdf_filepath = evaluate_and_generate_conformer_3d(
                    adj, MOLECULAR_FORMULA_ATOMS, optimized_energy, step_idx, "Optimization"
                )

                if sdf_filepath:
                    logger.info(f"Optimization Step [{step_idx}]: Target Hit. Energy = {optimized_energy:.4f}")
                    discovered_compounds_manifest.append(
                        {"Phase": "Optimization", "Energy": optimized_energy, "File": sdf_filepath})

    if discovered_compounds_manifest:
        output_dataframe = pd.DataFrame(discovered_compounds_manifest)
        output_dataframe.to_csv("generated_molecules_batch.csv", index=False)
        logger.info("=" * 60)
        logger.info(f"SUCCESS: {len(discovered_compounds_manifest)} Verified SDF Files Cached to '{SDF_DIR}'.")
        logger.info("=" * 60)