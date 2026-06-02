# -*- coding: utf-8 -*-
"""
Project: De Novo Molecular Design with Quantum-Classic Hybrid Framework
Description:
    This script implements a hybrid quantum-classical workflow for molecular design.
    It utilizes a coherent Ising machine (CIM) for combinatorial optimization
    constrained by valency rules, and a classical surrogate model for property guidance.

Key Optimizations:
    1. Augmented Ising Transformation: Handles linear terms via auxiliary spin.
    2. Precision Safety Margin: Scaling factor 0.95 to prevent hardware overflow.
    3. Data Quality Control: Filters high-energy samples during initialization.

Platform: Kaiwu SDK (CIM Backend)
"""

import time
import os
import shutil
import random
import numpy as np
import pandas as pd
import kaiwu as kw
import kaiwu.preprocess as kw_pre
from sklearn.linear_model import LinearRegression

# ==========================================
# Configuration & Initialization
# ==========================================
CHECKPOINT_DIR = './tmp_kaiwu_checkpoint'


def init_environment():
    """Initialize SDK environment and license."""
    if os.path.exists(CHECKPOINT_DIR):
        try:
            shutil.rmtree(CHECKPOINT_DIR)
        except Exception:
            pass
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    kw.common.CheckpointManager.save_dir = CHECKPOINT_DIR

    print("[INFO] Initializing Kaiwu SDK environment...")
    try:
        kw.license.init(
            user_id="Your_id",
            sdk_code="Your_code"
        )
        print("[INFO] License verified successfully.")
    except Exception as e:
        print(f"[WARNING] License initialization warning: {e}")


# ==========================================
# Core Logic: Molecule Designer
# ==========================================
class MolecularDesignStandard:
    """
    Manages the mapping from molecular graph to Ising model.
    Includes:
    - QUBO matrix construction (Objective + Constraints)
    - Augmented Ising transformation (Auxiliary spin for linear terms)
    - Precision adaptation for CIM hardware
    """

    def __init__(self, atoms, max_valencies):
        self.atoms = atoms
        self.max_valencies = max_valencies
        self.n_atoms = len(atoms)

        # Map bond variables to indices
        self.idx_counter = 0
        self.bond_indices = {}
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                self.bond_indices[(i, j)] = self.idx_counter
                self.idx_counter += 1

        # Map slack variables to indices
        self.slack_indices = {}
        for i in range(self.n_atoms):
            max_val = self.max_valencies[i]
            num_bits = int(np.floor(np.log2(max_val))) + 1
            for k in range(num_bits):
                self.slack_indices[(i, k)] = self.idx_counter
                self.idx_counter += 1

        self.num_qubo_vars = self.idx_counter
        # Add 1 auxiliary spin for Linear->Quadratic transformation
        self.total_ising_vars = self.num_qubo_vars + 1
        print(
            f"[INFO] Problem Size initialized. Logic Vars: {self.num_qubo_vars}, Total Ising Vars: {self.total_ising_vars}")

    def get_bond_index(self, u, v):
        if u < v:
            return self.bond_indices.get((u, v))
        else:
            return self.bond_indices.get((v, u))

    def build_qubo_matrix(self, surrogate_coeffs=None, penalty_strength=100.0):
        """Constructs the raw QUBO matrix with objective and constraints."""
        Q = np.zeros((self.num_qubo_vars, self.num_qubo_vars))

        # 1. Objective Term (Surrogate Model)
        feature_list = list(self.bond_indices.keys())
        if surrogate_coeffs is not None:
            for i, (u, v) in enumerate(feature_list):
                if i < len(surrogate_coeffs):
                    beta = float(surrogate_coeffs[i])
                    idx = self.bond_indices[(u, v)]
                    Q[idx, idx] += beta  # Linear terms on diagonal

        # 2. Constraint Term (Valency Constraints)
        for i in range(self.n_atoms):
            max_val = self.max_valencies[i]
            terms = []
            # Bond variables
            for j in range(self.n_atoms):
                if i == j: continue
                idx = self.get_bond_index(i, j)
                if idx is not None: terms.append((idx, 1.0))

            # Slack variables
            num_bits = int(np.floor(np.log2(max_val))) + 1
            for k in range(num_bits):
                idx = self.slack_indices[(i, k)]
                terms.append((idx, 2.0 ** k))

            constant = -float(max_val)

            # Expand (Sum - Max)^2
            for p in range(len(terms)):
                idx_p, coef_p = terms[p]
                linear_val = (coef_p ** 2) + (2 * constant * coef_p)
                Q[idx_p, idx_p] += penalty_strength * linear_val
                for q in range(p + 1, len(terms)):
                    idx_q, coef_q = terms[q]
                    quad_val = 2 * coef_p * coef_q
                    if idx_p < idx_q:
                        Q[idx_p, idx_q] += penalty_strength * quad_val
                    else:
                        Q[idx_q, idx_p] += penalty_strength * quad_val
        return Q

    def qubo_to_ising(self, Q_qubo):
        """
        Transforms QUBO to standard Ising matrix with zero diagonal.
        Introduces an auxiliary spin to handle linear terms.
        """
        N = self.num_qubo_vars
        J = np.zeros((N + 1, N + 1))

        # Symmetrize Q
        Q_sym = np.triu(Q_qubo) + np.triu(Q_qubo, 1).T
        h_vec = np.zeros(N)

        for i in range(N):
            h_vec[i] += Q_sym[i, i] / 2.0
            for j in range(i + 1, N):
                val = Q_sym[i, j] / 4.0
                J[i, j] = val
                J[j, i] = val
                h_vec[i] += val
                h_vec[j] += val

        # Embed linear fields into interactions with auxiliary spin
        aux_idx = N
        for i in range(N):
            J[i, aux_idx] = h_vec[i]
            J[aux_idx, i] = h_vec[i]

        return J

    def adapt_precision(self, J_matrix):
        """
        Adapts the Ising matrix for hardware precision limits.
        Strategy: Scaling with safety margin + Mutation.
        """
        max_abs = np.max(np.abs(J_matrix))
        if max_abs == 0: return J_matrix

        # [Optimization] Use safety margin (0.95) to prevent boundary overflow
        target_max = 100.0
        safety_margin = 0.95

        scale_factor = (target_max / max_abs) * safety_margin
        J_scaled = np.round(J_matrix * scale_factor)

        try:
            J_final = kw_pre.perform_precision_adaption_mutate(J_scaled)
            return J_final
        except Exception as e:
            # Fallback logic for robustness
            print(f"[WARNING] Mutate failed ({e}), employing manual fallback.")
            np.fill_diagonal(J_scaled, 0)
            return J_scaled

    def solve_on_cim(self, J_matrix, task_name, max_retries=3):
        """Submits the task to CIM optimizer with network retry logic."""
        # Backup matrix for audit
        pd.DataFrame(J_matrix).to_csv(f"{task_name}.csv", index=False, header=False)

        optimizer = kw.cim.CIMOptimizer(
            task_name_prefix=task_name,
            wait=True,
            interval=1
        )

        print(f"[INFO] Submitting task: {task_name} (Shape: {J_matrix.shape})")

        # Robust submission loop
        for attempt in range(max_retries):
            try:
                return optimizer.solve(J_matrix)
            except Exception as e:
                print(f"[WARNING] Connection glitch (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("[ERROR] All retries failed for this step.")
                    return []
                time.sleep(3)  # Wait before retry

    def parse_result(self, ising_sol):
        """
        Decodes Ising solution back to adjacency matrix.
        Handles auxiliary spin correction.
        """
        if len(ising_sol) != self.total_ising_vars:
            return None, None

        spins = ising_sol[:-1]
        aux_spin = ising_sol[-1]

        # Correct spin direction relative to auxiliary spin
        final_spins = spins * aux_spin
        binary_sol = (final_spins + 1) // 2

        adj_mat = np.zeros((self.n_atoms, self.n_atoms), dtype=int)
        for (u, v), idx in self.bond_indices.items():
            if idx < len(binary_sol):
                val = int(binary_sol[idx])
                adj_mat[u, v] = val
                adj_mat[v, u] = val

        feature_vec = []
        for (u, v) in self.bond_indices.keys():
            feature_vec.append(adj_mat[u, v])

        return adj_mat, feature_vec


# ==========================================
# Oracle / Energy Function
# ==========================================
def calculate_oracle_energy(adj_matrix, atoms):
    """Calculates the free energy of a given molecular structure."""
    energy = 0.0
    n = len(atoms)
    bond_energies = {('C', 'C'): -2.5, ('C', 'N'): -3.0, ('C', 'O'): -3.5, ('O', 'O'): 1.0}
    bonds_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] == 1:
                bonds_count += 1
                pair = tuple(sorted((atoms[i], atoms[j])))
                energy += bond_energies.get(pair, -0.5)

    # Stability penalty for disconnected structures
    if bonds_count < n - 1: energy += 5.0
    return energy + np.random.normal(0, 0.05)


def simulate_large_dataset(n_samples, atoms, max_valencies, designer):
    """
    Generates synthetic dataset for pre-training.
    NOTE: Added a High-Energy Filter to reject unstable samples.
    """
    print(f"[INFO] Generating {n_samples} synthetic samples (with High-Energy Filter)...")
    X_data, y_data = [], []
    n = len(atoms)
    valid_count = 0
    total_attempts = 0

    # Threshold for filtering out unstable structures during init
    ENERGY_THRESHOLD = 5.0

    while valid_count < n_samples:
        total_attempts += 1
        adj = np.zeros((n, n), dtype=int)
        degrees = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    if degrees[i] < max_valencies[i] and degrees[j] < max_valencies[j]:
                        adj[i, j] = 1
                        adj[j, i] = 1
                        degrees[i] += 1
                        degrees[j] += 1

        # Calculate energy
        energy = calculate_oracle_energy(adj, atoms)

        # [Filter] Discard samples with poor stability
        if energy > ENERGY_THRESHOLD:
            continue

        feat = []
        for (u, v) in designer.bond_indices.keys(): feat.append(adj[u, v])
        X_data.append(feat)
        y_data.append(energy)
        valid_count += 1

        if valid_count % 100 == 0:
            print(f"   ... Progress: {valid_count}/{n_samples}")

    print(f"[INFO] Dataset generation complete. Final size: {len(X_data)}")
    return X_data, y_data


# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == '__main__':
    # Initialize System
    init_environment()

    # ==========================================
    # [Config] Manual Hyperparameters
    # Tuned for C=6, O=2 system based on experimental runs
    # ==========================================
    ATOMS = ['C', 'C', 'C', 'C', 'C', 'C', 'O', 'O']
    MAX_VALENCIES = [4, 4, 4, 4, 4, 4, 2, 2]

    # Optimization settings
    N_SAMPLES = 500  # Sample size for regression stability
    PENALTY_EXPLORE = 80.0  # Lower penalty for broader exploration
    PENALTY_OPT = 120.0  # Higher penalty to enforce strict constraints

    print(f"[Config] System: {ATOMS}")
    print(f"[Config] Penalty Schedule: {PENALTY_EXPLORE} -> {PENALTY_OPT}")

    designer = MolecularDesignStandard(ATOMS, MAX_VALENCIES)

    # Step 1: Data Preparation
    dataset_X, dataset_y = simulate_large_dataset(N_SAMPLES, ATOMS, MAX_VALENCIES, designer)

    # Step 2: Phase 1 - Random Exploration
    print("\n" + "=" * 40)
    print("PHASE 1: CIM EXPLORATION")
    print("=" * 40)

    for i in range(2):
        Q_qubo = designer.build_qubo_matrix(surrogate_coeffs=None, penalty_strength=PENALTY_EXPLORE)
        J_ising = designer.qubo_to_ising(Q_qubo)
        J_final = designer.adapt_precision(J_ising)

        solutions = designer.solve_on_cim(J_final, task_name=f"explore_{i}_{int(time.time())}")

        if len(solutions) > 0:
            count = 0
            for sol in solutions:
                adj, feat = designer.parse_result(sol)
                if adj is not None:
                    energy = calculate_oracle_energy(adj, ATOMS)
                    dataset_X.append(feat)
                    dataset_y.append(energy)
                    count += 1
                if count >= 3: break
            print(f"[INFO] Batch {i}: {count} valid samples added.")

    # Step 3: Phase 2 - Surrogate Optimization
    print("\n" + "=" * 40)
    print(f"PHASE 2: SURROGATE OPTIMIZATION (Dataset: {len(dataset_X)})")
    print("=" * 40)

    regressor = LinearRegression()

    for step in range(3):
        print(f"\n--- Optimization Step {step + 1} ---")

        # Train Surrogate
        regressor.fit(dataset_X, dataset_y)
        beta_coeffs = regressor.coef_
        print(f"[Model] Surrogate Coefficients Mean: {np.mean(beta_coeffs):.3f}")

        # Build & Solve
        Q_qubo = designer.build_qubo_matrix(surrogate_coeffs=beta_coeffs, penalty_strength=PENALTY_OPT)
        J_ising = designer.qubo_to_ising(Q_qubo)
        J_final = designer.adapt_precision(J_ising)

        solutions = designer.solve_on_cim(J_final, task_name=f"opt_{step}_{int(time.time())}")

        # Feedback Loop
        if len(solutions) > 0:
            best_sol = solutions[0]
            adj, feat = designer.parse_result(best_sol)
            if adj is not None:
                current_energy = calculate_oracle_energy(adj, ATOMS)
                dataset_X.append(feat)
                dataset_y.append(current_energy)
                print(f"[RESULT] Found Molecule Energy: {current_energy:.4f}")
            else:
                print("[WARNING] Invalid solution format received.")
        else:
            print("[WARNING] No feasible solution found.")

    print("\n[INFO] Workflow completed successfully.")