import json
import logging
from pathlib import Path

import polars as pl
from pymer4.models import lmer


def perform_repeated_measures_analysis(
    wave: str = "baseline_year_1_arm_1",
    experiment_number: int = 1,
    version_name: str = "test",
    predictor: str = "class_label",
):
    # Define the brain modalities
    modalities = [
        "bilateral_cortical_thickness",
        "bilateral_cortical_volume",
        "bilateral_cortical_surface_area",
        "bilateral_subcortical_volume",
        "bilateral_tract_FA",
        "bilateral_tract_MD",
    ]

    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    if data_store_path.exists():
        print("Mounted data store path: ", data_store_path)
        logging.info("Mounted data store path: %s", data_store_path)

    analysis_root_path = Path(
        data_store_path,
        "users",
        "Eric",
        "depression_trajectories",
    )

    processed_data_path = Path(
        analysis_root_path,
        version_name,
        "processed_data",
    )

    # File paths
    features_path = Path(
        processed_data_path,
        f"mri_all_features_with_traj_long_rescaled-{wave}.csv",
    )

    feature_sets_path = Path(
        processed_data_path,
        "features_of_interest.json",
    )

    # Load imaging and covariate data
    features_df = pl.read_csv(
        features_path,
        has_header=True,
    )

    # Load feature sets for each modality
    with open(feature_sets_path, "r") as f:
        feature_sets = json.load(f)

    # Convert categorical variables
    categorical_cols = [
        "src_subject_id",
        "class_label",
        "demo_sex_v2",
        "img_device_label",
        "hemisphere",
        "site_id_l",
        "rel_family_id",
        "demo_comb_income_v2",
    ]

    features_df = features_df.with_columns(
        [pl.col(col).cast(pl.String).cast(pl.Categorical) for col in categorical_cols]
    )

    # Effects of interest
    effects_of_interest = [
        "class_label1",
        "class_label2",
        "class_label3",
        "hemisphereRight:class_label1",
        "hemisphereRight:class_label2",
        "hemisphereRight:class_label3",
    ]

    # Store results here
    results_list = []

    # Loop over each modality
    for modality in modalities:
        logging.info(f"Fitting LMEs for: {modality}")

        roi_list = feature_sets[modality]

        # Fixed effects to include
        fixed_effects = [
            "interview_age",
            "age2",
            "demo_sex_v2",
            "img_device_label",
            "demo_comb_income_v2",
            "BMI_zscore",
        ]

        if modality == "bilateral_cortical_thickness":
            fixed_effects.append("smri_thick_cdk_mean")

        elif modality == "bilateral_cortical_surface_area":
            fixed_effects.append("smri_area_cdk_total")

        elif modality == "bilateral_cortical_volume":
            fixed_effects.append("smri_vol_scs_intracranialv")

        elif modality == "bilateral_subcortical_volume":
            fixed_effects.append("smri_vol_scs_intracranialv")

        elif modality == "bilateral_tract_FA":
            fixed_effects.append("FA_all_dti_atlas_tract_fibers")

        elif modality == "bilateral_tract_MD":
            fixed_effects.append("MD_all_dti_atlas_tract_fibers")

        for feature in roi_list:
            logging.info(f"Fitting model for: {feature}")

            # The formula estimates a random intercept for each subject
            # and family nested within sites
            formula = f"{feature} ~ hemisphere * {predictor} + {' + '.join(fixed_effects)} + (1|src_subject_id) + (1|site_id_l:rel_family_id)"  # noqa: E501
            try:
                model = lmer(
                    formula,
                    data=features_df,
                )

                model.fit(summarize=False)

                # Check convergence status
                if "FALSE" in str(model.convergence_status):
                    logging.warning(
                        f"Model did not converge for {feature} in {modality} for {wave}"
                    )
                    print(f"Convergence failed: {feature}")

            except Exception as e:
                print(f"Model failed for {feature} in {modality} for {wave}: {e}")
                logging.error(
                    f"Model failed for {feature} in {modality} for {wave}: {e}"
                )
                continue

            # Save both main trajectory class effect and hemisphere interaction
            model_stats = model.result_fit

            # Filter by effect names (assuming first column contains effect names)
            coefs = model_stats.filter(
                pl.col(model_stats.columns[0]).is_in(effects_of_interest)
            ).with_columns(
                [pl.lit(modality).alias("modality"), pl.lit(feature).alias("feature")]
            )

            results_list.append(coefs)

    # Create results directory
    results_path = Path(
        analysis_root_path,
        version_name,
        f"exp_{experiment_number}",
    )

    results_path.mkdir(parents=True, exist_ok=True)

    # Save results as CSV
    results_df = pl.concat(results_list)
    results_df.write_csv(
        Path(
            results_path,
            f"repeated_bilateral_traj_results-{wave}.csv",
        )
    )

    logging.info(
        f"Repeated analysis complete for {wave}. Results saved to: {results_path / f'repeated_bilateral_traj_results-{wave}.csv'}"  # noqa: E501
    )


if __name__ == "__main__":
    wave = "2_year_follow_up_y_arm_1"

    perform_repeated_measures_analysis(wave=wave)
