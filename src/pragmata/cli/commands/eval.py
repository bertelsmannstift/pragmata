"""CLI commands for evaluation workflows."""

import typer

from pragmata import eval
from pragmata.api import UNSET
from pragmata.cli.parsing import parse_cli_value

eval_app = typer.Typer(help="Evaluation commands.")


@eval_app.command("train-evaluator")
def train_evaluator_command(
    labeled_data_path: str | None = typer.Option(
        None,
        "--labeled-data-path",
        help="Path to labeled eval training CSV. If omitted, an annotation export is used.",
    ),
    export_id: str | None = typer.Option(
        None,
        "--export-id",
        help="Annotation export identifier. Ignored when --labeled-data-path is provided.",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help="Eval task to train for: retrieval, grounding, or generation.",
    ),
    base_dir: str | None = typer.Option(
        None,
        "--base-dir",
        help="Workspace base directory. Defaults to current working directory.",
    ),
    config_path: str | None = typer.Option(
        None,
        "--config",
        help="Path to the config file.",
    ),
    target_name: str | None = typer.Option(
        None,
        "--target-name",
        help="Display name passed to tlmtc logs and reports.",
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help="Target Hugging Face checkpoint used for final fine-tuning.",
    ),
    proxy_checkpoint: str | None = typer.Option(
        None,
        "--proxy-checkpoint",
        help="Proxy Hugging Face checkpoint used for hyperparameter tuning.",
    ),
    scale_learning_rate: bool | None = typer.Option(
        None,
        "--scale-learning-rate/--no-scale-learning-rate",
        help="Enable or disable tlmtc learning-rate scaling between proxy and target checkpoints.",
    ),
    sequence_length: int | None = typer.Option(
        None,
        "--sequence-length",
        help="Maximum combined tokenized sequence length for text and text_pair.",
    ),
    train_kwargs: str | None = typer.Option(
        None,
        "--train-kwargs",
        help="Additional tlmtc train_tlmtc keyword arguments. Accepts a JSON object.",
    ),
) -> None:
    """Train a supervised evaluator model."""
    result = eval.train_evaluator(
        labeled_data_path=UNSET if labeled_data_path is None else labeled_data_path,
        export_id=UNSET if export_id is None else export_id,
        task=UNSET if task is None else task,
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config_path is None else config_path,
        target_name=UNSET if target_name is None else target_name,
        checkpoint=UNSET if checkpoint is None else checkpoint,
        proxy_checkpoint=UNSET if proxy_checkpoint is None else proxy_checkpoint,
        scale_learning_rate=UNSET if scale_learning_rate is None else scale_learning_rate,
        sequence_length=UNSET if sequence_length is None else sequence_length,
        train_kwargs=parse_cli_value(train_kwargs),
    )

    typer.echo("Evaluator training run complete.")
    typer.echo(f"run_id: {result.paths.run_id}")
    typer.echo(f"run_directory: {result.paths.run_dir}")
    typer.echo(f"model_directory: {result.paths.model_dir}")


@eval_app.command("predict-labels")
def predict_command(
    unlabeled_data_path: str | None = typer.Option(
        None,
        "--unlabeled-data-path",
        help="Path to the task-specific unlabeled eval prediction CSV.",
    ),
    evaluator_run_id: str | None = typer.Option(
        None,
        "--evaluator-run-id",
        help="Evaluator training run identifier. If omitted, the latest task-compatible evaluator is used.",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help="Eval task to predict for: retrieval, grounding, or generation.",
    ),
    base_dir: str | None = typer.Option(
        None,
        "--base-dir",
        help="Workspace base directory. Defaults to current working directory.",
    ),
    config_path: str | None = typer.Option(
        None,
        "--config",
        help="Path to the config file.",
    ),
    predict_kwargs: str | None = typer.Option(
        None,
        "--predict-kwargs",
        help="Additional tlmtc predict_tlmtc keyword arguments. Accepts a JSON object.",
    ),
) -> None:
    """Predict evaluation labels with a trained evaluator."""
    result = eval.predict_labels(
        unlabeled_data_path=UNSET if unlabeled_data_path is None else unlabeled_data_path,
        evaluator_run_id=UNSET if evaluator_run_id is None else evaluator_run_id,
        task=UNSET if task is None else task,
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config_path is None else config_path,
        predict_kwargs=parse_cli_value(predict_kwargs),
    )

    typer.echo("Evaluator prediction run complete.")
    typer.echo(f"evaluator_run_id: {result.paths.run_id}")
    typer.echo(f"prediction_directory: {result.paths.prediction_run_dir}")
    typer.echo(f"probabilities: {result.paths.probabilities_path}")
    typer.echo(f"predictions: {result.paths.predictions_path}")
