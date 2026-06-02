# -*- coding: utf-8 -*-
"""
Project: Quantum-Classical Hybrid Framework - Property Distribution Analysis
Version: 2.0 (Publication-Quality Rendering Edition)

Description:
    This analytical module performs quantitative evaluation of generated molecular candidates.
    It computes critical physicochemical properties (QED and LogP) using the RDKit engine,
    and constructs publication-ready Kernel Density Estimation (KDE) comparative distributions
    against the baseline training set statistics reported in the original literature.

Features:
    - Automated SDF parsing and physicochemical property computation.
    - Synthetic Gaussian distribution generation based on paper baseline statistics.
    - Publication-ready dual-axis KDE visualization with rug plots.

Author: Zhong Xuan (Hunan Agricultural University / National University of Defense Technology)
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chemical Informatics Engine
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')  # Suppress C++ backend warnings for clean output
except ImportError:
    raise ImportError("Critical Dependency Error: 'rdkit' is required for property computation.")

# =========================================================================
# System Configuration & Logging Setup
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("PropertyAnalyzer")

SDF_DIRECTORY: str = './batch_molecules_sdf'
OUTPUT_FIGURE_NAME: str = 'reproduced_figure_4_high_res.png'

# Literature Baseline Statistics (Source: Training Set benchmark)
# "The averages of QED... and LogP... are 0.728 and 2.450, and standard deviations are 0.138 and 1.433."
BASELINE_STATS = {
    'QED': {'mean': 0.728, 'std': 0.138},
    'LogP': {'mean': 2.450, 'std': 1.433}
}


# =========================================================================
# Data Processing Modules
# =========================================================================
def evaluate_molecular_properties(sdf_dir: str) -> pd.DataFrame:
    """
    Parses structural data files (.sdf) and computes pharmacokinetic properties.
    """
    extracted_data = []
    target_path = Path(sdf_dir)

    if not target_path.exists() or not target_path.is_dir():
        logger.error(f"Target directory '{sdf_dir}' not found. Please execute the generation sequence first.")
        return pd.DataFrame()

    sdf_files = list(target_path.glob('*.sdf'))
    logger.info(f"Detected {len(sdf_files)} generated 3D molecular candidates.")

    for file_path in sdf_files:
        try:
            supplier = Chem.SDMolSupplier(str(file_path))
            for mol in supplier:
                if mol is not None:
                    calc_qed = QED.qed(mol)
                    calc_logp = Descriptors.MolLogP(mol)

                    # Distinguish operational phases based on file signature
                    if "Optimization" in file_path.name:
                        phase_label = "QC (Optimized)"
                    elif "Exploration" in file_path.name:
                        phase_label = "QC (Initial Seeds)"
                    else:
                        phase_label = "QC (Unknown)"

                    extracted_data.append({
                        'Source': 'QC Generated (Ours)',
                        'Phase': phase_label,
                        'QED': calc_qed,
                        'LogP': calc_logp
                    })
        except Exception as e:
            logger.warning(f"Parse failure on {file_path.name}. Integrity compromised: {e}")

    return pd.DataFrame(extracted_data)


def synthesize_baseline_distribution(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generates a synthetic dataset mimicking the literature's training set
    using Gaussian distribution sampling based on reported statistics.
    """
    logger.info(f"Synthesizing baseline distribution ({n_samples} samples) from literature statistics...")

    # QED is strictly bounded between [0, 1]
    qed_synthetic = np.random.normal(BASELINE_STATS['QED']['mean'], BASELINE_STATS['QED']['std'], n_samples)
    qed_synthetic = np.clip(qed_synthetic, 0.0, 1.0)

    logp_synthetic = np.random.normal(BASELINE_STATS['LogP']['mean'], BASELINE_STATS['LogP']['std'], n_samples)

    synthetic_df = pd.DataFrame({
        'Source': ['Training Set (Baseline)'] * n_samples,
        'Phase': ['Baseline'] * n_samples,
        'QED': qed_synthetic,
        'LogP': logp_synthetic
    })

    return synthetic_df


# =========================================================================
# Visualization Rendering Engine
# =========================================================================
def render_property_distributions(df_combined: pd.DataFrame, df_quantum: pd.DataFrame) -> None:
    """
    Renders a publication-ready dual-axis figure comparing QED and LogP distributions.
    Applies strict academic aesthetics (white background, explicit spines).
    """
    logger.info("Initializing high-resolution matplotlib rendering engine...")

    # Academic Plotting Aesthetics Setup
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    color_palette = {"Training Set (Baseline)": "#95a5a6", "QC Generated (Ours)": "#3498db"}

    # ---------------------------------------------------------
    # Panel (a): Quantitative Estimation of Drug-likeness (QED)
    # ---------------------------------------------------------
    sns.kdeplot(
        data=df_combined, x="QED", hue="Source",
        fill=True, common_norm=False, palette=color_palette,
        alpha=0.4, linewidth=2, ax=axes[0], legend=False
    )
    # Rug plot representing precise discrete points of generated hits
    sns.rugplot(data=df_quantum, x="QED", color="#e74c3c", height=0.08, ax=axes[0], linewidth=1.5)

    axes[0].set_title("Quantitative Estimation of Drug-likeness (QED)", fontweight='bold', pad=15)
    axes[0].set_xlabel("QED Value (Higher is better)", labelpad=10)
    axes[0].set_ylabel("Density", labelpad=10)
    axes[0].set_xlim(0.0, 1.0)

    # ---------------------------------------------------------
    # Panel (b): Wildman-Crippen Partition Coefficient (LogP)
    # ---------------------------------------------------------
    sns.kdeplot(
        data=df_combined, x="LogP", hue="Source",
        fill=True, common_norm=False, palette=color_palette,
        alpha=0.4, linewidth=2, ax=axes[1]
    )
    sns.rugplot(data=df_quantum, x="LogP", color="#e74c3c", height=0.08, ax=axes[1], linewidth=1.5)

    axes[1].set_title("Partition Coefficient (LogP)", fontweight='bold', pad=15)
    axes[1].set_xlabel("LogP Value", labelpad=10)
    axes[1].set_ylabel("Density", labelpad=10)
    axes[1].set_xlim(-3.0, 6.0)

    # Clean up aesthetics
    sns.despine(fig=fig, offset=10, trim=False)
    sns.move_legend(axes[1], "upper right", bbox_to_anchor=(1.0, 1.0), title=None, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_NAME, dpi=300, bbox_inches='tight')
    logger.info(f"Figure successfully rendered and saved to '{OUTPUT_FIGURE_NAME}'")


# =========================================================================
# Main Execution Pipeline
# =========================================================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("STARTING MOLECULAR PROPERTY ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # 1. Extract and compute properties from Quantum Hardware output
    df_quantum = evaluate_molecular_properties(SDF_DIRECTORY)

    if df_quantum.empty:
        logger.error("Pipeline Terminated: Missing molecular data. Execute generative sequence first.")
    else:
        # 2. Synthesize comparative baseline data
        df_baseline = synthesize_baseline_distribution()

        # 3. Aggregate datasets
        df_combined = pd.concat([df_baseline, df_quantum], ignore_index=True)

        # 4. Render and export figure
        render_property_distributions(df_combined, df_quantum)

        # 5. Output Analytical Briefing
        logger.info("\n" + "=" * 40)
        logger.info("PROPERTY STATISTICAL BRIEFING (MEAN VALUES)")
        logger.info("=" * 40)
        stats_summary = df_quantum.groupby('Phase')[['QED', 'LogP']].mean().round(3)
        print(stats_summary.to_string())
        logger.info("=" * 40)