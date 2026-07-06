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
    trust_remote_code: bool | None = typer.Option(
        None,
        "--trust-remote-code/--no-trust-remote-code",
        help="Allow or disallow Hugging Face custom checkpoint code execution.",
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
        trust_remote_code=UNSET if trust_remote_code is None else trust_remote_code,
        train_kwargs=parse_cli_value(train_kwargs),
    )

    typer.echo("Evaluator training run complete.")
    typer.echo(f"run_id: {result.paths.run_id}")
    typer.echo(f"run_directory: {result.paths.run_dir}")
    typer.echo(f"model_directory: {result.paths.model_dir}")
