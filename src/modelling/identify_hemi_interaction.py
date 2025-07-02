import json
import logging
from pathlib import Path

import polars as pl
from statsmodels.stats.multitest import multipletests


def identify_sig_inter_terms(
    wave: str = "baseline_year_1_arm_1",
    experiment_number: int = 1,
    version_name: str = "test",
):
    """Identify significant hemisphere interaction terms.

    Applies FDR correction separately for each modality and each interaction term.
    """
    modalities = [
        "bilateral_cortical_thickness",
        "bilateral_cortical_volume",
        "bilateral_cortical_surface_area",
        "bilateral_subcortical_volume",
        "bilateral_tract_FA",
        "bilateral_tract_MD",
    ]

    # Match path structure from repeated_measures.py
    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    if data_store_path.exists():
        logging.info("Mounted data store path: %s", data_store_path)

    analysis_root_path = Path(
        data_store_path,
        "users",
        "Eric",
        "depression_trajectories",
    )

    experiments_path = Path(
        analysis_root_path,
        version_name,
        f"exp_{experiment_number}",
    )

    results_path = Path(
        experiments_path,
        "results",
    )

    # Load results using polars
    repeated_results = pl.read_csv(
        Path(
            results_path,
            f"repeated_bilateral_traj_results-{wave}.csv",
        )
    )

    significant_features_by_modality = {}

    # The interaction terms of interest (as in repeated_measures.py)
    interaction_terms = [
        "hemisphereRight:class_label1",
        "hemisphereRight:class_label2",
        "hemisphereRight:class_label3",
    ]

    # Apply FDR separately for each modality AND each interaction term
    for modality in modalities:
        logging.info(f"Processing modality: {modality}")

        modality_df = repeated_results.filter(
            (pl.col("modality") == modality)
            & (pl.col("effect_name").is_in(interaction_terms))
        )

        if modality_df.height > 0:
            significant_features_by_interaction = {}

            # Apply FDR separately for each interaction term within this modality
            for interaction_term in interaction_terms:
                interaction_df = modality_df.filter(
                    pl.col("effect_name") == interaction_term
                )

                if interaction_df.height > 0:
                    # Extract p-values and apply FDR correction
                    pvals = interaction_df.select("P-val").to_numpy().flatten()

                    rejected, qvals, _, _ = multipletests(
                        pvals, alpha=0.05, method="fdr_bh"
                    )

                    # Add FDR results back to the dataframe
                    interaction_df = interaction_df.with_columns(
                        [
                            pl.Series("q_value", qvals),
                            pl.Series("significant", rejected),
                        ]
                    )

                    # Filter significant features
                    significant = interaction_df.filter(pl.col("significant"))

                    if significant.height > 0:
                        significant_features_by_interaction[interaction_term] = (
                            significant.select("feature").to_series().to_list()
                        )

                        feature_list = significant_features_by_interaction[
                            interaction_term
                        ]
                        logging.info(f"{modality} - {interaction_term}: {feature_list}")

            if significant_features_by_interaction:
                significant_features_by_modality[modality] = (
                    significant_features_by_interaction
                )

    # Save results to json
    results_path.mkdir(parents=True, exist_ok=True)

    with open(
        Path(
            results_path,
            f"sig_interaction_terms-{wave}.json",
        ),
        "w",
    ) as f:
        json.dump(significant_features_by_modality, f, indent=4)

    logging.info(
        f"Significant interaction analysis complete for {wave}. "
        f"Results saved to: {results_path / f'sig_interaction_terms-{wave}.json'}"
    )


if __name__ == "__main__":
    wave = "baseline_year_1_arm_1"
    version_name = "test"
    experiment_number = 1

    identify_sig_inter_terms(
        wave=wave,
        version_name=version_name,
        experiment_number=experiment_number,
    )
