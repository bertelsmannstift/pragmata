"""Input dataframe contracts for eval workflows."""

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from pragmata.core.schemas.annotation_task import Task


class EvalInputSchemaError(ValueError):
    """Raised when eval input data violates the expected dataframe contract."""


TEXT_COLUMNS_BY_TASK: dict[Task, tuple[str, ...]] = {
    Task.RETRIEVAL: ("query", "chunk"),
    Task.GROUNDING: ("answer", "context_set"),
    Task.GENERATION: ("query", "answer"),
}


LABEL_COLUMNS_BY_TASK: dict[Task, tuple[str, ...]] = {
    Task.RETRIEVAL: (
        "topically_relevant",
        "evidence_sufficient",
        "misleading",
    ),
    Task.GROUNDING: (
        "support_present",
        "unsupported_claim_present",
        "contradicted_claim_present",
        "source_cited",
        "fabricated_source",
    ),
    Task.GENERATION: (
        "proper_action",
        "response_on_topic",
        "helpful",
        "incomplete",
        "unsafe_content",
    ),
}


_NON_BLANK_STRING = pa.Check(
    lambda series: series.str.strip().ne(""),
    error="must not contain blank strings",
)


_BINARY_LABEL = pa.Check.isin([0, 1])


# Ranks are 1-based; K (the cutoff) is inferred from the data at scoring time,
# so only a positive lower bound is enforced here.
_POSITIVE_RANK = pa.Check.greater_than_or_equal_to(1)


_NON_EMPTY_FRAME = pa.Check(
    lambda df: len(df) > 0,
    error="dataframe must contain at least one row",
)


def _text_column_schemas(
    task: Task,
) -> dict[str, pa.Column]:
    return {
        column: pa.Column(
            str,
            nullable=False,
            required=True,
            checks=_NON_BLANK_STRING,
        )
        for column in TEXT_COLUMNS_BY_TASK[task]
    }


def _label_column_schemas(
    task: Task,
) -> dict[str, pa.Column]:
    return {
        column: pa.Column(
            int,
            nullable=False,
            required=True,
            coerce=True,
            checks=_BINARY_LABEL,
        )
        for column in LABEL_COLUMNS_BY_TASK[task]
    }


def _identity_column_schemas(
    task: Task,
) -> dict[str, pa.Column]:
    """Structural grouping/identity columns that scoring requires but training does not.

    Every task needs ``record_uuid`` to group rows into the per-query unit that
    metrics average over. Retrieval additionally needs ``chunk_id`` (stable
    per-chunk identity within a query) and ``chunk_rank`` (the 1-based rank
    within the query, required by NDCG@K and MRR@K).
    """
    columns: dict[str, pa.Column] = {
        "record_uuid": pa.Column(str, nullable=False, required=True, checks=_NON_BLANK_STRING),
    }
    if task == Task.RETRIEVAL:
        columns["chunk_id"] = pa.Column(str, nullable=False, required=True, checks=_NON_BLANK_STRING)
        columns["chunk_rank"] = pa.Column(int, nullable=False, required=True, coerce=True, checks=_POSITIVE_RANK)
    return columns


RETRIEVAL_TRAIN_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.RETRIEVAL),
        **_label_column_schemas(Task.RETRIEVAL),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


GROUNDING_TRAIN_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.GROUNDING),
        **_label_column_schemas(Task.GROUNDING),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


GENERATION_TRAIN_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.GENERATION),
        **_label_column_schemas(Task.GENERATION),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


RETRIEVAL_PREDICT_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.RETRIEVAL),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


GROUNDING_PREDICT_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.GROUNDING),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


GENERATION_PREDICT_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.GENERATION),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


RETRIEVAL_SCORE_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.RETRIEVAL),
        **_label_column_schemas(Task.RETRIEVAL),
        **_identity_column_schemas(Task.RETRIEVAL),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


GROUNDING_SCORE_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.GROUNDING),
        **_label_column_schemas(Task.GROUNDING),
        **_identity_column_schemas(Task.GROUNDING),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


GENERATION_SCORE_SCHEMA = pa.DataFrameSchema(
    {
        **_text_column_schemas(Task.GENERATION),
        **_label_column_schemas(Task.GENERATION),
        **_identity_column_schemas(Task.GENERATION),
    },
    checks=[_NON_EMPTY_FRAME],
    strict=False,
    ordered=False,
    coerce=False,
)


_TRAIN_SCHEMAS_BY_TASK: dict[Task, pa.DataFrameSchema] = {
    Task.RETRIEVAL: RETRIEVAL_TRAIN_SCHEMA,
    Task.GROUNDING: GROUNDING_TRAIN_SCHEMA,
    Task.GENERATION: GENERATION_TRAIN_SCHEMA,
}


_SCORE_SCHEMAS_BY_TASK: dict[Task, pa.DataFrameSchema] = {
    Task.RETRIEVAL: RETRIEVAL_SCORE_SCHEMA,
    Task.GROUNDING: GROUNDING_SCORE_SCHEMA,
    Task.GENERATION: GENERATION_SCORE_SCHEMA,
}


_PREDICT_SCHEMAS_BY_TASK: dict[Task, pa.DataFrameSchema] = {
    Task.RETRIEVAL: RETRIEVAL_PREDICT_SCHEMA,
    Task.GROUNDING: GROUNDING_PREDICT_SCHEMA,
    Task.GENERATION: GENERATION_PREDICT_SCHEMA,
}


def validate_eval_train_frame(
    df: pd.DataFrame,
    *,
    task: Task,
) -> pd.DataFrame:
    """Validate a labeled eval training dataframe.

    Args:
        df: Input dataframe to validate.
        task: Annotation task that determines the input contract to apply.

    Returns:
        Validated dataframe with all original columns preserved.

    Raises:
        EvalInputSchemaError: If the input is not a dataframe or violates the
            task-specific training contract.
    """
    if not isinstance(df, pd.DataFrame):
        raise EvalInputSchemaError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    try:
        return _TRAIN_SCHEMAS_BY_TASK[task].validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise EvalInputSchemaError("Input dataframe violates the eval training data contract.") from exc


def validate_eval_score_frame(
    df: pd.DataFrame,
    *,
    task: Task,
) -> pd.DataFrame:
    """Validate a labeled eval scoring dataframe.

    Scoring has stricter structural requirements than training: every task must
    carry ``record_uuid`` (the per-query grouping unit), and retrieval must also
    carry ``chunk_id`` and ``chunk_rank``. These are enforced by dedicated
    ``*_SCORE_SCHEMA`` definitions rather than the training schemas, so training
    inputs are not constrained by scoring-only metadata.

    Args:
        df: Input dataframe to validate.
        task: Annotation task that determines the input contract to apply.

    Returns:
        Validated dataframe with all original columns preserved.

    Raises:
        EvalInputSchemaError: If the input is not a dataframe or violates the
            task-specific scoring contract.
    """
    if not isinstance(df, pd.DataFrame):
        raise EvalInputSchemaError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    try:
        return _SCORE_SCHEMAS_BY_TASK[task].validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise EvalInputSchemaError("Input dataframe violates the eval scoring data contract.") from exc


def validate_eval_predict_frame(
    df: pd.DataFrame,
    *,
    task: Task,
) -> pd.DataFrame:
    """Validate an unlabeled eval prediction dataframe.

    Args:
        df: Input dataframe to validate.
        task: Annotation task that determines the input contract to apply.

    Returns:
        Validated dataframe with all original columns preserved.

    Raises:
        EvalInputSchemaError: If the input is not a dataframe, contains label
            columns for the selected task, contains generic `label_*` columns, or
            violates the task-specific prediction contract.
    """
    if not isinstance(df, pd.DataFrame):
        raise EvalInputSchemaError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    try:
        validated = _PREDICT_SCHEMAS_BY_TASK[task].validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise EvalInputSchemaError("Input dataframe violates the eval prediction data contract.") from exc

    forbidden_label_cols = [
        column
        for column in validated.columns
        if column in LABEL_COLUMNS_BY_TASK[task] or str(column).startswith("label_")
    ]
    if forbidden_label_cols:
        raise EvalInputSchemaError(
            f"Prediction input must be unlabeled for task {task.value}, "
            f"but found label columns: {forbidden_label_cols}."
        )

    return validated
