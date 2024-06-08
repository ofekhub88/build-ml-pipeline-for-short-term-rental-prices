#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info(f"Download Artifact {args.input_artifact} from W&B ")
    run = wandb.init(project="nyc_airbnb", group="eda", save_code=True)
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info("preforme basic cleanup ")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info(f"Save the cleaned file in W&B {args.output_artifact}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_name = args.output_artifact
        temp_path = os.path.join(tmp_dir, artifact_name)
        df.to_csv(temp_path, index=False)
        artifact = wandb.Artifact(
            name=artifact_name,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(temp_path)
        run.log_artifact(artifact)
        # This waits for the artifact to be uploaded to W&B. If you
        # do not add this, the temp directory might be removed before
        # W&B had a chance to upload the datasets, and the upload
        # might fail
        artifact.wait()
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully qualified name for the artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the W&B artifact that will be created",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of file for the W&B artifact that will be created",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price", type=float, help="Minumum price for outliers data", required=True
    )

    parser.add_argument(
        "--max_price", type=float, help="Maximum price for outliers data", required=True
    )

    args = parser.parse_args()

    go(args)
