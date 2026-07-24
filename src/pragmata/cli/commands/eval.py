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


@eval_app.command("score")
def score_command(
    task: str | None = typer.Option(
        None,
        "--task",
        help="Eval task to score: retrieval, grounding, or generation.",
    ),
    path: str | None = typer.Option(
        None,
        "--path",
        help="Direct path to the labeled CSV to score.",
    ),
    export_id: str | None = typer.Option(
        None,
        "--export-id",
        help="Annotation export identifier; resolves to the task-specific exported CSV.",
    ),
    prediction_id: str | None = typer.Option(
        None,
        "--prediction-id",
        help="Prediction run identifier. Not yet supported (lands with eval predict).",
    ),
    score_id: str | None = typer.Option(
        None,
        "--score-id",
        help="Output identifier naming eval/scores/<score-id>/. Defaults to a generated value.",
    ),
    base_dir: str | None = typer.Option(
        None,
        "--base-dir",
        help="Workspace base directory. Defaults to current working directory.",
    ),
    n_resamples: int | None = typer.Option(
        None,
        "--n-resamples",
        help="Bootstrap iterations for the continuous metrics' confidence intervals.",
    ),
    ci: float | None = typer.Option(
        None,
        "--ci",
        help="Confidence level for every reported interval (e.g. 0.95).",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="RNG seed for reproducible bootstrap intervals.",
    ),
    config_path: str | None = typer.Option(
        None,
        "--config",
        help="Path to the config file.",
    ),
) -> None:
    """Score labeled eval data into corpus metrics with confidence intervals.

    The input is selected by one of ``--path`` / ``--export-id`` /
    ``--prediction-id`` (precedence in that order); with none, the latest
    annotation export is used.
    """
    report = eval.score(
        task=UNSET if task is None else task,
        path=UNSET if path is None else path,
        export_id=UNSET if export_id is None else export_id,
        prediction_id=UNSET if prediction_id is None else prediction_id,
        score_id=UNSET if score_id is None else score_id,
        base_dir=UNSET if base_dir is None else base_dir,
        n_resamples=UNSET if n_resamples is None else n_resamples,
        ci=UNSET if ci is None else ci,
        seed=UNSET if seed is None else seed,
        config_path=UNSET if config_path is None else config_path,
    )

    typer.echo(f"\n{report.task.value} scores (n={report.n_examples}, {report.ci_level:.0%} CI):")
    for name, metric in report.metric_scores():
        if metric is None:
            typer.echo(f"  {name}: n/a (not computed)")
        else:
            typer.echo(
                f"  {name}: {metric.point:.3f} "
                f"[{metric.ci_lower:.3f}, {metric.ci_upper:.3f}] ({metric.method}, n={metric.n})"
            )
