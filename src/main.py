import logging
from pathlib import Path

from preprocess.prepare_data import preprocess


def main(
    wave: str,
    version_name: str,
):
    """Main function to run the entire analysis pipeline."""
    # Call the preprocess function with the specified parameters
    preprocess(
        wave=wave,
        version_name=version_name,
    )


if __name__ == "__main__":
    # Define the wave and version name
    wave = "baseline_year_1_arm_1"
    version_name = "test"

    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    analysis_root_path = Path(
        data_store_path,
        "users",
        "Eric",
        "depression_trajectories",
    )

    version_path = Path(
        analysis_root_path,
        version_name,
    )
    version_path.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = version_path / "experiment.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also log to console
        ],
    )

    logging.info("Starting experiment with version: %s", version_name)
    logging.info("Log file saved to: %s", log_file)

    # Run the main function
    main(
        wave=wave,
        version_name=version_name,
    )
