"""Compile a Kubeflow Pipelines v2 workflow for Iris training and evaluation."""

import argparse
import os
from pathlib import Path

from kfp import compiler, dsl

DEFAULT_IMAGE = os.getenv("IRIS_TRAINING_IMAGE", "iris-ml:local")


@dsl.container_component
def train_and_evaluate(
    output_model: dsl.Output[dsl.Model],
    output_metrics: dsl.Output[dsl.Metrics],
    c: float = 1.0,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["python", "-m", "src.pipeline_task"],
        args=[
            "--c",
            c,
            "--model-output",
            output_model.path,
            "--metrics-output",
            output_metrics.path,
        ],
    )


@dsl.pipeline(
    name="beginner-iris-training",
    description="Train and evaluate a small Iris logistic-regression model.",
)
def iris_training_pipeline(c: float = 1.0):
    task = train_and_evaluate(c=c)
    task.set_cpu_request("100m")
    task.set_cpu_limit("500m")
    task.set_memory_request("256Mi")
    task.set_memory_limit("512Mi")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pipelines/iris-training-pipeline.yaml"),
        help="Where to write the compiled KFP v2 pipeline package.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(iris_training_pipeline, str(args.output))
    print(f"Compiled Kubeflow pipeline to {args.output}")


if __name__ == "__main__":
    main()
