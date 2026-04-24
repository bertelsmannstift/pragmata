"""Tests for the query generation API orchestration."""

from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent

import pytest

import pragmata.api.querygen as querygen_api
from pragmata.core.paths.querygen_paths import QueryGenRunPaths
from pragmata.core.schemas.querygen_output import PlanningSummaryArtifact, SyntheticQueriesMeta, SyntheticQueryRow
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_realize import RealizedQuery
from pragmata.core.schemas.querygen_summary import PlanningSummaryState

pytestmark = pytest.mark.usefixtures("workflow_stubs")


@pytest.fixture(autouse=True)
def mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure an API key is always present for orchestration tests."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-secret")


@pytest.fixture
def workflow_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install default workflow stubs so tests stay unit-level and deterministic."""
    _install_default_workflow_stubs(monkeypatch)


def _required_querygen_kwargs(tmp_path: Path) -> dict[str, object]:
    """Return the minimal valid kwargs for gen_queries()."""
    return {
        "domains": "public administration",
        "roles": "policy analyst",
        "languages": "en",
        "topics": "digital services",
        "intents": "learn",
        "tasks": "summarize",
        "base_dir": tmp_path,
        "model_provider": "mistralai",
    }


def _make_blueprint(
    candidate_id: str,
    *,
    domain: str = "public administration",
    role: str = "policy analyst",
    language: str = "en",
    topic: str = "digital services",
    intent: str = "learn",
    task: str = "summarize",
    difficulty: str | None = "basic",
    format: str | None = "bullet list",
) -> QueryBlueprint:
    """Build a valid QueryBlueprint for tests."""
    return QueryBlueprint(
        candidate_id=candidate_id,
        domain=domain,
        role=role,
        language=language,
        topic=topic,
        intent=intent,
        task=task,
        difficulty=difficulty,
        format=format,
        user_scenario=f"Scenario for {candidate_id}",
        information_need=f"Information need for {candidate_id}",
    )


def _make_realized_query(
    candidate_id: str,
    *,
    query: str | None = None,
) -> RealizedQuery:
    """Build a valid RealizedQuery for tests."""
    return RealizedQuery(
        candidate_id=candidate_id,
        query=query or f"Realized query for {candidate_id}",
    )


def _make_row(
    *,
    query_id: str,
    query: str = "How do I access this service?",
    domain: str | None = "public administration",
    role: str | None = "policy analyst",
    language: str | None = "en",
    topic: str | None = "digital services",
    intent: str | None = "learn",
    task: str | None = "summarize",
    difficulty: str | None = "basic",
    format: str | None = "bullet list",
) -> SyntheticQueryRow:
    """Build a valid SyntheticQueryRow for tests."""
    return SyntheticQueryRow(
        query_id=query_id,
        query=query,
        domain=domain,
        role=role,
        language=language,
        topic=topic,
        intent=intent,
        task=task,
        difficulty=difficulty,
        format=format,
    )


def _make_meta(
    *,
    run_id: str,
    n_requested_queries: int,
    n_returned_queries: int,
    model_provider: str = "mistralai",
    planning_model: str = "magistral-medium-latest",
    realization_model: str = "mistral-medium-latest",
) -> SyntheticQueriesMeta:
    """Build valid SyntheticQueriesMeta for tests."""
    return SyntheticQueriesMeta(
        run_id=run_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        n_requested_queries=n_requested_queries,
        n_returned_queries=n_returned_queries,
        model_provider=model_provider,
        planning_model=planning_model,
        realization_model=realization_model,
    )


def _make_planning_summary_state(
    *,
    redundancy_patterns: str = "Repeated service-access questions for the same administrative scenario.",
    diversification_targets: str = "Broaden toward comparison, explanation, and eligibility scenarios.",
    coverage_notes: str = "Basic public-service access requests are already well covered.",
) -> PlanningSummaryState:
    """Build a valid PlanningSummaryState for tests."""
    return PlanningSummaryState(
        redundancy_patterns=redundancy_patterns,
        diversification_targets=diversification_targets,
        coverage_notes=coverage_notes,
    )


def _make_planning_summary_artifact(
    *,
    spec_fingerprint: str = "spec-fingerprint-123",
    source_run_id: str = "prior-run",
    created_at: datetime | None = None,
    state: PlanningSummaryState | None = None,
) -> PlanningSummaryArtifact:
    """Build a valid PlanningSummaryArtifact for tests."""
    return PlanningSummaryArtifact(
        spec_fingerprint=spec_fingerprint,
        source_run_id=source_run_id,
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
        state=state or _make_planning_summary_state(),
    )


def _install_default_workflow_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install default passthrough stubs for the staged workflow."""

    def build_candidate_ids(n_queries: int) -> list[str]:
        digits = max(3, len(str(n_queries)))
        return [f"c{index:0{digits}d}" for index in range(1, n_queries + 1)]

    def iter_batch_sizes(n_queries: int, batch_size: int):  # noqa: ANN202
        del batch_size
        return iter([n_queries])

    def read_planning_summary_artifact(
        *,
        artifact_path: Path,
        spec,  # noqa: ANN001
    ) -> PlanningSummaryArtifact | None:
        del artifact_path, spec
        return None

    def run_planning_stage(
        *,
        spec,  # noqa: ANN001
        llm_settings,  # noqa: ANN001
        api_key: str,
        batch_candidate_ids: list[str],
        planning_summary: PlanningSummaryState | None = None,
    ) -> list[QueryBlueprint]:
        del spec, llm_settings, api_key, planning_summary
        return [_make_blueprint(candidate_id) for candidate_id in batch_candidate_ids]

    def run_planning_summary(
        *,
        spec,  # noqa: ANN001
        candidates: list[QueryBlueprint],
        llm_settings,  # noqa: ANN001
        api_key: str,
        prior_summary_state: PlanningSummaryState | None = None,
    ) -> PlanningSummaryState:
        del spec, llm_settings, api_key, prior_summary_state
        first_candidate_id = candidates[0].candidate_id
        return _make_planning_summary_state(
            redundancy_patterns=f"Repeated patterns around {first_candidate_id}.",
            diversification_targets=f"Diversify beyond {first_candidate_id}.",
            coverage_notes=f"Coverage includes {first_candidate_id}.",
        )

    def filter_aligned_candidate_ids(
        items: list[object],
        expected_candidate_ids: list[str],
    ) -> list[object]:
        del expected_candidate_ids
        return items

    def deduplicate_blueprints(
        candidates: list[QueryBlueprint],
        near_duplicate_tolerance: float = 0.95,
    ) -> list[QueryBlueprint]:
        del near_duplicate_tolerance
        return candidates

    def chunk_blueprints(
        blueprints: list[QueryBlueprint],
        chunk_size: int,
    ):  # noqa: ANN202
        del chunk_size
        return iter([blueprints] if blueprints else [])

    def run_realization_stage(
        *,
        candidates: list[QueryBlueprint],
        llm_settings,  # noqa: ANN001
        api_key: str,
    ) -> list[RealizedQuery]:
        del llm_settings, api_key
        return [_make_realized_query(candidate.candidate_id) for candidate in candidates]

    def assemble_query_rows(
        *,
        blueprints: list[QueryBlueprint],
        realized_queries: list[RealizedQuery],
        run_id: str,
    ) -> list[SyntheticQueryRow]:
        del blueprints
        return [
            _make_row(
                query_id=f"{run_id}_q{index:03d}",
                query=realized_query.query,
            )
            for index, realized_query in enumerate(realized_queries, start=1)
        ]

    def assemble_queries_meta(
        *,
        run_id: str,
        n_requested_queries: int,
        n_returned_queries: int,
        model_provider: str,
        planning_model: str,
        realization_model: str,
    ) -> SyntheticQueriesMeta:
        return _make_meta(
            run_id=run_id,
            n_requested_queries=n_requested_queries,
            n_returned_queries=n_returned_queries,
            model_provider=model_provider,
            planning_model=planning_model,
            realization_model=realization_model,
        )

    def assemble_planning_summary(
        *,
        spec,  # noqa: ANN001
        run_id: str,
        state: PlanningSummaryState,
    ) -> PlanningSummaryArtifact:
        del spec
        return _make_planning_summary_artifact(
            source_run_id=run_id,
            state=state,
        )

    def export_queries(
        *,
        rows: list[SyntheticQueryRow],
        meta: SyntheticQueriesMeta,
        queries_path: Path,
        meta_path: Path,
    ) -> None:
        del rows, meta, queries_path, meta_path

    def export_planning_summary(
        *,
        artifact: PlanningSummaryArtifact,
        artifact_path: Path,
    ) -> None:
        del artifact, artifact_path

    monkeypatch.setattr(querygen_api, "build_candidate_ids", build_candidate_ids)
    monkeypatch.setattr(querygen_api, "iter_batch_sizes", iter_batch_sizes)
    monkeypatch.setattr(querygen_api, "read_planning_summary_artifact", read_planning_summary_artifact)
    monkeypatch.setattr(querygen_api, "run_planning_stage", run_planning_stage)
    monkeypatch.setattr(querygen_api, "run_planning_summary", run_planning_summary)
    monkeypatch.setattr(querygen_api, "filter_aligned_candidate_ids", filter_aligned_candidate_ids)
    monkeypatch.setattr(querygen_api, "deduplicate_blueprints", deduplicate_blueprints)
    monkeypatch.setattr(querygen_api, "chunk_blueprints", chunk_blueprints)
    monkeypatch.setattr(querygen_api, "run_realization_stage", run_realization_stage)
    monkeypatch.setattr(querygen_api, "assemble_query_rows", assemble_query_rows)
    monkeypatch.setattr(querygen_api, "assemble_queries_meta", assemble_queries_meta)
    monkeypatch.setattr(querygen_api, "assemble_planning_summary", assemble_planning_summary)
    monkeypatch.setattr(querygen_api, "export_queries", export_queries)
    monkeypatch.setattr(querygen_api, "export_planning_summary", export_planning_summary)


def test_gen_queries_combines_user_args_config_and_defaults(tmp_path: Path) -> None:
    """gen_queries combines user args, config values, and model defaults."""
    config_path = tmp_path / "querygen.yml"
    config_path.write_text(
        dedent(
            """\
            llm:
              model_provider: mistralai
              planning_model: custom-planner
            n_queries: 10
            batch_size: 12
            run_id: original-id
            """
        ),
        encoding="utf-8",
    )

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        config_path=config_path,
        run_id="overridden-id",
    )

    assert result.settings.n_queries == 10
    assert result.settings.llm.planning_model == "custom-planner"
    assert result.settings.run_id == "overridden-id"
    assert result.paths.run_dir.name == result.settings.run_id
    assert result.settings.llm.realization_model == "mistral-medium-latest"
    assert result.settings.batch_size == 12
    assert result.settings.near_duplicate_tolerance == 0.95
    assert result.settings.enable_planning_memory is True


def test_gen_queries_orchestrates_run_paths(tmp_path: Path) -> None:
    """gen_queries resolves and creates the expected run path scaffold."""
    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        run_id="run-123",
    )

    expected_tool_root = tmp_path.resolve() / "querygen"
    expected_run_dir = expected_tool_root / "runs" / "run-123"

    assert result.paths.tool_root == expected_tool_root
    assert result.paths.run_dir == expected_run_dir
    assert result.paths.synthetic_queries_csv == expected_run_dir / "synthetic_queries.csv"
    assert result.paths.synthetic_queries_meta_json == expected_run_dir / "synthetic_queries.meta.json"
    assert result.paths.planning_summary_artifact_json.parent == expected_tool_root
    assert result.paths.planning_summary_artifact_json.suffix == ".json"
    assert expected_run_dir.is_dir()


def test_gen_queries_fingerprints_spec_before_resolving_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gen_queries fingerprints the resolved spec before querygen paths are resolved."""
    call_order: list[str] = []
    seen: dict[str, object] = {}

    def fingerprint_querygen_spec(spec):  # noqa: ANN001
        call_order.append("fingerprint")
        seen["fingerprint_spec"] = spec
        return "spec-fingerprint-123"

    def resolve_querygen_paths(
        *,
        workspace,  # noqa: ANN001
        run_id: str,
        spec_fingerprint: str,
    ) -> QueryGenRunPaths:
        call_order.append("resolve_paths")
        seen["workspace_base_dir"] = workspace.base_dir
        seen["run_id"] = run_id
        seen["spec_fingerprint"] = spec_fingerprint

        tool_root = tmp_path.resolve() / "querygen"
        run_dir = tool_root / "runs" / run_id

        return QueryGenRunPaths(
            tool_root=tool_root,
            run_dir=run_dir,
            synthetic_queries_csv=run_dir / "synthetic_queries.csv",
            synthetic_queries_meta_json=run_dir / "synthetic_queries.meta.json",
            planning_summary_artifact_json=tool_root / f"{spec_fingerprint}.json",
        )

    monkeypatch.setattr(querygen_api, "fingerprint_querygen_spec", fingerprint_querygen_spec)
    monkeypatch.setattr(querygen_api, "resolve_querygen_paths", resolve_querygen_paths)

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        run_id="fingerprint-order-check",
    )

    assert call_order == ["fingerprint", "resolve_paths"]
    assert seen["run_id"] == "fingerprint-order-check"
    assert seen["workspace_base_dir"] == tmp_path.resolve()
    assert seen["spec_fingerprint"] == "spec-fingerprint-123"
    assert seen["fingerprint_spec"] == result.settings.spec
    assert result.paths.planning_summary_artifact_json == (
        tmp_path.resolve() / "querygen" / "spec-fingerprint-123.json"
    )


def test_gen_queries_reads_planning_summary_artifact_from_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gen_queries loads the initial planning-summary artifact from the reusable artifact path."""
    read_calls: list[dict[str, object]] = []

    def read_planning_summary_artifact(
        *,
        artifact_path: Path,
        spec,  # noqa: ANN001
    ) -> PlanningSummaryArtifact | None:
        read_calls.append(
            {
                "artifact_path": artifact_path,
                "spec": spec,
            }
        )
        return None

    monkeypatch.setattr(querygen_api, "read_planning_summary_artifact", read_planning_summary_artifact)

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        run_id="planning-summary-read-check",
    )

    assert read_calls == [
        {
            "artifact_path": result.paths.planning_summary_artifact_json,
            "spec": result.settings.spec,
        }
    ]


def test_gen_queries_returns_result_object(tmp_path: Path) -> None:
    """gen_queries returns the structured run result."""
    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        run_id="result-check",
    )

    assert isinstance(result, querygen_api.QueryGenRunResult)
    assert result.settings.run_id == "result-check"
    assert result.paths.run_dir.name == "result-check"


@pytest.mark.parametrize("batch_size", [1, 7, 25])
def test_gen_queries_accepts_batch_size_override(
    tmp_path: Path,
    batch_size: int,
) -> None:
    """gen_queries accepts explicit batch_size overrides."""
    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        batch_size=batch_size,
        run_id=f"batch-size-check-{batch_size}",
    )

    assert result.settings.batch_size == batch_size


@pytest.mark.parametrize(
    ("config_batch_size", "arg_batch_size"),
    [(12, 7), (25, 1), (3, 3)],
)
def test_gen_queries_batch_size_arg_overrides_config_value(
    tmp_path: Path,
    config_batch_size: int,
    arg_batch_size: int,
) -> None:
    """Explicit batch_size arg takes precedence over config batch_size."""
    config_path = tmp_path / "querygen.yml"
    config_path.write_text(
        dedent(
            f"""\
            llm:
              model_provider: mistralai
            batch_size: {config_batch_size}
            """
        ),
        encoding="utf-8",
    )

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        config_path=config_path,
        batch_size=arg_batch_size,
        run_id="batch-size-precedence-check",
    )

    assert result.settings.batch_size == arg_batch_size


def test_gen_queries_repeats_planning_batches_and_applies_stage1_filter_to_aggregated_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning batching repeats in order and stage-1 filtering sees aggregated outputs."""
    candidate_ids = ["c001", "c002", "c003", "c004", "c005"]
    planning_batch_calls: list[list[str]] = []
    stage1_filter_seen: list[dict[str, list[str]]] = []

    monkeypatch.setattr(querygen_api, "build_candidate_ids", lambda n_queries: candidate_ids)
    monkeypatch.setattr(querygen_api, "iter_batch_sizes", lambda n_queries, batch_size: iter([2, 2, 1]))

    def run_planning_stage(
        *,
        spec,  # noqa: ANN001
        llm_settings,  # noqa: ANN001
        api_key: str,
        batch_candidate_ids: list[str],
        planning_summary: PlanningSummaryState | None = None,
    ) -> list[QueryBlueprint]:
        del spec, llm_settings, api_key, planning_summary
        planning_batch_calls.append(list(batch_candidate_ids))
        return [_make_blueprint(candidate_id) for candidate_id in batch_candidate_ids]

    def filter_aligned_candidate_ids(
        items: list[object],
        expected_candidate_ids: list[str],
    ) -> list[object]:
        if items and isinstance(items[0], QueryBlueprint):
            stage1_filter_seen.append(
                {
                    "item_ids": [item.candidate_id for item in items],
                    "expected_ids": expected_candidate_ids,
                }
            )
        return items

    monkeypatch.setattr(querygen_api, "run_planning_stage", run_planning_stage)
    monkeypatch.setattr(querygen_api, "filter_aligned_candidate_ids", filter_aligned_candidate_ids)

    querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=5,
        batch_size=2,
        run_id="planning-batch-order-check",
    )

    assert planning_batch_calls == [
        ["c001", "c002"],
        ["c003", "c004"],
        ["c005"],
    ]
    assert stage1_filter_seen == [
        {
            "item_ids": ["c001", "c002", "c003", "c004", "c005"],
            "expected_ids": ["c001", "c002", "c003", "c004", "c005"],
        }
    ]


def test_gen_queries_invokes_first_planning_batch_with_none_summary_when_no_prior_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no prior artifact is available, the first planning batch starts with no planning summary."""
    planning_summary_seen: list[PlanningSummaryState | None] = []

    monkeypatch.setattr(querygen_api, "read_planning_summary_artifact", lambda **kwargs: None)

    def run_planning_stage(
        *,
        spec,  # noqa: ANN001
        llm_settings,  # noqa: ANN001
        api_key: str,
        batch_candidate_ids: list[str],
        planning_summary: PlanningSummaryState | None = None,
    ) -> list[QueryBlueprint]:
        del spec, llm_settings, api_key
        planning_summary_seen.append(planning_summary)
        return [_make_blueprint(candidate_id) for candidate_id in batch_candidate_ids]

    monkeypatch.setattr(querygen_api, "run_planning_stage", run_planning_stage)

    querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=3,
        run_id="first-planning-summary-none-check",
    )

    assert planning_summary_seen == [None]


def test_gen_queries_threads_planning_summary_state_across_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning-summary state is updated after each batch and threaded into the next planning batch."""
    candidate_ids = ["c001", "c002", "c003", "c004", "c005"]
    planning_batch_calls: list[dict[str, object]] = []
    planning_summary_calls: list[dict[str, object]] = []

    summary_state_1 = _make_planning_summary_state(
        redundancy_patterns="summary-state-1 redundancy",
        diversification_targets="summary-state-1 targets",
        coverage_notes="summary-state-1 coverage",
    )
    summary_state_2 = _make_planning_summary_state(
        redundancy_patterns="summary-state-2 redundancy",
        diversification_targets="summary-state-2 targets",
        coverage_notes="summary-state-2 coverage",
    )
    summary_state_3 = _make_planning_summary_state(
        redundancy_patterns="summary-state-3 redundancy",
        diversification_targets="summary-state-3 targets",
        coverage_notes="summary-state-3 coverage",
    )
    returned_states = iter([summary_state_1, summary_state_2, summary_state_3])

    monkeypatch.setattr(querygen_api, "build_candidate_ids", lambda n_queries: candidate_ids)
    monkeypatch.setattr(querygen_api, "iter_batch_sizes", lambda n_queries, batch_size: iter([2, 2, 1]))
    monkeypatch.setattr(querygen_api, "read_planning_summary_artifact", lambda **kwargs: None)

    def run_planning_stage(
        *,
        spec,  # noqa: ANN001
        llm_settings,  # noqa: ANN001
        api_key: str,
        batch_candidate_ids: list[str],
        planning_summary: PlanningSummaryState | None = None,
    ) -> list[QueryBlueprint]:
        del spec, llm_settings, api_key
        planning_batch_calls.append(
            {
                "batch_candidate_ids": list(batch_candidate_ids),
                "planning_summary": planning_summary,
            }
        )
        return [_make_blueprint(candidate_id) for candidate_id in batch_candidate_ids]

    def run_planning_summary(
        *,
        spec,  # noqa: ANN001
        candidates: list[QueryBlueprint],
        llm_settings,  # noqa: ANN001
        api_key: str,
        prior_summary_state: PlanningSummaryState | None = None,
    ) -> PlanningSummaryState:
        del spec, llm_settings, api_key
        planning_summary_calls.append(
            {
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
                "prior_summary_state": prior_summary_state,
            }
        )
        return next(returned_states)

    monkeypatch.setattr(querygen_api, "run_planning_stage", run_planning_stage)
    monkeypatch.setattr(querygen_api, "run_planning_summary", run_planning_summary)

    querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=5,
        batch_size=2,
        run_id="planning-summary-threading-check",
    )

    assert planning_batch_calls == [
        {
            "batch_candidate_ids": ["c001", "c002"],
            "planning_summary": None,
        },
        {
            "batch_candidate_ids": ["c003", "c004"],
            "planning_summary": summary_state_1,
        },
        {
            "batch_candidate_ids": ["c005"],
            "planning_summary": summary_state_2,
        },
    ]
    assert planning_summary_calls == [
        {
            "candidate_ids": ["c001", "c002"],
            "prior_summary_state": None,
        },
        {
            "candidate_ids": ["c003", "c004"],
            "prior_summary_state": summary_state_1,
        },
        {
            "candidate_ids": ["c005"],
            "prior_summary_state": summary_state_2,
        },
    ]


def test_gen_queries_applies_stage1_filtering_before_deduplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-1 filtering is applied before deduplication."""
    call_order: list[str] = []

    def filter_aligned_candidate_ids(
        items: list[object],
        expected_candidate_ids: list[str],
    ) -> list[object]:
        del expected_candidate_ids
        if items and isinstance(items[0], QueryBlueprint):
            call_order.append("stage1_filter")
            return [items[0], items[2]]
        call_order.append("stage2_filter")
        return items

    def deduplicate_blueprints(
        candidates: list[QueryBlueprint],
        near_duplicate_tolerance: float = 0.95,
    ) -> list[QueryBlueprint]:
        del near_duplicate_tolerance
        call_order.append("deduplicate")
        assert [candidate.candidate_id for candidate in candidates] == ["c001", "c003"]
        return candidates

    monkeypatch.setattr(querygen_api, "filter_aligned_candidate_ids", filter_aligned_candidate_ids)
    monkeypatch.setattr(querygen_api, "deduplicate_blueprints", deduplicate_blueprints)

    querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=3,
        run_id="stage1-filter-before-dedup-check",
    )

    assert call_order.index("stage1_filter") < call_order.index("deduplicate")


def test_gen_queries_drives_realization_batches_from_selected_blueprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realization-stage batching is driven from the selected blueprints."""
    selected_blueprints = [
        _make_blueprint("c001"),
        _make_blueprint("c003"),
        _make_blueprint("c004"),
    ]
    realization_batch_calls: list[list[str]] = []

    def deduplicate_blueprints(
        candidates: list[QueryBlueprint],
        near_duplicate_tolerance: float = 0.95,
    ) -> list[QueryBlueprint]:
        del candidates, near_duplicate_tolerance
        return selected_blueprints

    def chunk_blueprints(
        blueprints: list[QueryBlueprint],
        chunk_size: int,
    ):  # noqa: ANN202
        assert blueprints == selected_blueprints
        assert chunk_size == 2
        return iter([blueprints[:2], blueprints[2:]])

    def run_realization_stage(
        *,
        candidates: list[QueryBlueprint],
        llm_settings,  # noqa: ANN001
        api_key: str,
    ) -> list[RealizedQuery]:
        del llm_settings, api_key
        realization_batch_calls.append([candidate.candidate_id for candidate in candidates])
        return [_make_realized_query(candidate.candidate_id) for candidate in candidates]

    monkeypatch.setattr(querygen_api, "deduplicate_blueprints", deduplicate_blueprints)
    monkeypatch.setattr(querygen_api, "chunk_blueprints", chunk_blueprints)
    monkeypatch.setattr(querygen_api, "run_realization_stage", run_realization_stage)

    querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=4,
        batch_size=2,
        run_id="realization-batch-source-check",
    )

    assert realization_batch_calls == [
        ["c001", "c003"],
        ["c004"],
    ]


def test_gen_queries_applies_stage2_filtering_before_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-2 filtering is applied before assembly using selected blueprint IDs."""
    selected_blueprints = [
        _make_blueprint("c002"),
        _make_blueprint("c001"),
        _make_blueprint("c003"),
    ]
    call_order: list[str] = []

    def deduplicate_blueprints(
        candidates: list[QueryBlueprint],
        near_duplicate_tolerance: float = 0.95,
    ) -> list[QueryBlueprint]:
        del candidates, near_duplicate_tolerance
        return selected_blueprints

    def filter_aligned_candidate_ids(
        items: list[object],
        expected_candidate_ids: list[str],
    ) -> list[object]:
        if items and isinstance(items[0], QueryBlueprint):
            return items

        call_order.append("stage2_filter")
        assert expected_candidate_ids == ["c002", "c001", "c003"]
        realized_items = items
        assert [item.candidate_id for item in realized_items] == ["c002", "c001", "c003"]
        return [realized_items[0], realized_items[2]]

    def assemble_query_rows(
        *,
        blueprints: list[QueryBlueprint],
        realized_queries: list[RealizedQuery],
        run_id: str,
    ) -> list[SyntheticQueryRow]:
        del run_id
        call_order.append("assemble")
        assert [blueprint.candidate_id for blueprint in blueprints] == ["c002", "c001", "c003"]
        assert [query.candidate_id for query in realized_queries] == ["c002", "c003"]
        return [
            _make_row(query_id="stage2-filter-check_q001", query="query 1"),
            _make_row(query_id="stage2-filter-check_q002", query="query 2"),
        ]

    monkeypatch.setattr(querygen_api, "deduplicate_blueprints", deduplicate_blueprints)
    monkeypatch.setattr(querygen_api, "filter_aligned_candidate_ids", filter_aligned_candidate_ids)
    monkeypatch.setattr(querygen_api, "assemble_query_rows", assemble_query_rows)

    querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=3,
        run_id="stage2-filter-check",
    )

    assert call_order == ["stage2_filter", "assemble"]


def test_gen_queries_calls_assembly_and_export_with_final_post_filter_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assembly and export receive the final post-filter stage outputs."""
    export_calls: list[dict[str, object]] = []
    selected_blueprints = [
        _make_blueprint("c001"),
        _make_blueprint("c003"),
    ]
    filtered_realization_outputs = [
        _make_realized_query("c001", query="Final realized query"),
    ]
    assembled_rows = [
        _make_row(
            query_id="assembly-export-check_q001",
            query="Final realized query",
        )
    ]
    assembled_meta = _make_meta(
        run_id="assembly-export-check",
        n_requested_queries=3,
        n_returned_queries=1,
    )
    assemble_rows_calls: list[dict[str, object]] = []
    assemble_meta_calls: list[dict[str, object]] = []

    monkeypatch.setattr(querygen_api, "build_candidate_ids", lambda n_queries: ["c001", "c002", "c003"])
    monkeypatch.setattr(querygen_api, "iter_batch_sizes", lambda n_queries, batch_size: iter([3]))

    def run_planning_stage(
        *,
        spec,  # noqa: ANN001
        llm_settings,  # noqa: ANN001
        api_key: str,
        batch_candidate_ids: list[str],
        planning_summary: PlanningSummaryState | None = None,
    ) -> list[QueryBlueprint]:
        del spec, llm_settings, api_key, planning_summary
        return [_make_blueprint(candidate_id) for candidate_id in batch_candidate_ids]

    def filter_aligned_candidate_ids(
        items: list[object],
        expected_candidate_ids: list[str],
    ) -> list[object]:
        del expected_candidate_ids
        if items and isinstance(items[0], QueryBlueprint):
            return items
        return filtered_realization_outputs

    def deduplicate_blueprints(
        candidates: list[QueryBlueprint],
        near_duplicate_tolerance: float = 0.95,
    ) -> list[QueryBlueprint]:
        del candidates, near_duplicate_tolerance
        return selected_blueprints

    def chunk_blueprints(
        blueprints: list[QueryBlueprint],
        chunk_size: int,
    ):  # noqa: ANN202
        del chunk_size
        assert blueprints == selected_blueprints
        return iter([blueprints])

    def run_realization_stage(
        *,
        candidates: list[QueryBlueprint],
        llm_settings,  # noqa: ANN001
        api_key: str,
    ) -> list[RealizedQuery]:
        del llm_settings, api_key
        assert candidates == selected_blueprints
        return [
            _make_realized_query("c001", query="Final realized query"),
            _make_realized_query("c003", query="Dropped by stage-2 filter"),
        ]

    def assemble_query_rows(
        *,
        blueprints: list[QueryBlueprint],
        realized_queries: list[RealizedQuery],
        run_id: str,
    ) -> list[SyntheticQueryRow]:
        assemble_rows_calls.append(
            {
                "blueprints": blueprints,
                "realized_queries": realized_queries,
                "run_id": run_id,
            }
        )
        return assembled_rows

    def assemble_queries_meta(
        *,
        run_id: str,
        n_requested_queries: int,
        n_returned_queries: int,
        model_provider: str,
        planning_model: str,
        realization_model: str,
    ) -> SyntheticQueriesMeta:
        assemble_meta_calls.append(
            {
                "run_id": run_id,
                "n_requested_queries": n_requested_queries,
                "n_returned_queries": n_returned_queries,
                "model_provider": model_provider,
                "planning_model": planning_model,
                "realization_model": realization_model,
            }
        )
        return assembled_meta

    def export_queries(
        *,
        rows: list[SyntheticQueryRow],
        meta: SyntheticQueriesMeta,
        queries_path: Path,
        meta_path: Path,
    ) -> None:
        export_calls.append(
            {
                "rows": rows,
                "meta": meta,
                "queries_path": queries_path,
                "meta_path": meta_path,
            }
        )

    monkeypatch.setattr(querygen_api, "run_planning_stage", run_planning_stage)
    monkeypatch.setattr(querygen_api, "filter_aligned_candidate_ids", filter_aligned_candidate_ids)
    monkeypatch.setattr(querygen_api, "deduplicate_blueprints", deduplicate_blueprints)
    monkeypatch.setattr(querygen_api, "chunk_blueprints", chunk_blueprints)
    monkeypatch.setattr(querygen_api, "run_realization_stage", run_realization_stage)
    monkeypatch.setattr(querygen_api, "assemble_query_rows", assemble_query_rows)
    monkeypatch.setattr(querygen_api, "assemble_queries_meta", assemble_queries_meta)
    monkeypatch.setattr(querygen_api, "export_queries", export_queries)

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=3,
        run_id="assembly-export-check",
    )

    assert assemble_rows_calls == [
        {
            "blueprints": selected_blueprints,
            "realized_queries": filtered_realization_outputs,
            "run_id": "assembly-export-check",
        }
    ]
    assert assemble_meta_calls == [
        {
            "run_id": "assembly-export-check",
            "n_requested_queries": 3,
            "n_returned_queries": 1,
            "model_provider": "mistralai",
            "planning_model": "magistral-medium-latest",
            "realization_model": "mistral-medium-latest",
        }
    ]
    assert export_calls == [
        {
            "rows": assembled_rows,
            "meta": assembled_meta,
            "queries_path": result.paths.synthetic_queries_csv,
            "meta_path": result.paths.synthetic_queries_meta_json,
        }
    ]


def test_gen_queries_assembles_and_exports_final_planning_summary_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final planning-summary assembly and export run once at the end of a successful run."""
    summary_state_1 = _make_planning_summary_state(
        redundancy_patterns="state-1 redundancy",
        diversification_targets="state-1 targets",
        coverage_notes="state-1 coverage",
    )
    summary_state_2 = _make_planning_summary_state(
        redundancy_patterns="state-2 redundancy",
        diversification_targets="state-2 targets",
        coverage_notes="state-2 coverage",
    )
    assembled_artifact = _make_planning_summary_artifact(
        spec_fingerprint="final-spec-fingerprint",
        source_run_id="planning-summary-export-check",
        state=summary_state_2,
    )

    monkeypatch.setattr(querygen_api, "build_candidate_ids", lambda n_queries: ["c001", "c002", "c003"])
    monkeypatch.setattr(querygen_api, "iter_batch_sizes", lambda n_queries, batch_size: iter([2, 1]))
    monkeypatch.setattr(querygen_api, "read_planning_summary_artifact", lambda **kwargs: None)

    returned_states = iter([summary_state_1, summary_state_2])

    def run_planning_summary(
        *,
        spec,  # noqa: ANN001
        candidates: list[QueryBlueprint],
        llm_settings,  # noqa: ANN001
        api_key: str,
        prior_summary_state: PlanningSummaryState | None = None,
    ) -> PlanningSummaryState:
        del spec, candidates, llm_settings, api_key, prior_summary_state
        return next(returned_states)

    assemble_calls: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []

    def assemble_planning_summary(
        *,
        spec,  # noqa: ANN001
        run_id: str,
        state: PlanningSummaryState,
    ) -> PlanningSummaryArtifact:
        assemble_calls.append(
            {
                "spec": spec,
                "run_id": run_id,
                "state": state,
            }
        )
        return assembled_artifact

    def export_planning_summary(
        *,
        artifact: PlanningSummaryArtifact,
        artifact_path: Path,
    ) -> None:
        export_calls.append(
            {
                "artifact": artifact,
                "artifact_path": artifact_path,
            }
        )

    monkeypatch.setattr(querygen_api, "run_planning_summary", run_planning_summary)
    monkeypatch.setattr(querygen_api, "assemble_planning_summary", assemble_planning_summary)
    monkeypatch.setattr(querygen_api, "export_planning_summary", export_planning_summary)

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=3,
        batch_size=2,
        run_id="planning-summary-export-check",
    )

    assert len(assemble_calls) == 1
    assert assemble_calls[0]["run_id"] == "planning-summary-export-check"
    assert assemble_calls[0]["state"] == summary_state_2
    assert assemble_calls[0]["spec"] == result.settings.spec

    assert export_calls == [
        {
            "artifact": assembled_artifact,
            "artifact_path": result.paths.planning_summary_artifact_json,
        }
    ]


def test_gen_queries_raises_when_provider_secret_resolution_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider-secret resolution failures stop the workflow before stage execution."""
    monkeypatch.setattr(
        querygen_api,
        "resolve_api_key",
        lambda provider: (_ for _ in ()).throw(RuntimeError("missing secret")),
    )
    monkeypatch.setattr(
        querygen_api,
        "build_candidate_ids",
        lambda n_queries: pytest.fail("build_candidate_ids must not run when secret resolution fails"),
    )

    with pytest.raises(RuntimeError, match="missing secret"):
        querygen_api.gen_queries(
            **_required_querygen_kwargs(tmp_path),
            run_id="missing-secret-check",
        )


def test_gen_queries_propagates_planning_failure_and_stops_downstream_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning failures propagate and prevent realization, assembly, and export."""
    monkeypatch.setattr(
        querygen_api,
        "run_planning_stage",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("planning failed")),
    )
    monkeypatch.setattr(
        querygen_api,
        "run_realization_stage",
        lambda **kwargs: pytest.fail("run_realization_stage must not run after planning failure"),
    )
    monkeypatch.setattr(
        querygen_api,
        "assemble_query_rows",
        lambda **kwargs: pytest.fail("assemble_query_rows must not run after planning failure"),
    )
    monkeypatch.setattr(
        querygen_api,
        "export_queries",
        lambda **kwargs: pytest.fail("export_queries must not run after planning failure"),
    )
    monkeypatch.setattr(
        querygen_api,
        "export_planning_summary",
        lambda **kwargs: pytest.fail("export_planning_summary must not run after planning failure"),
    )

    with pytest.raises(RuntimeError, match="planning failed"):
        querygen_api.gen_queries(
            **_required_querygen_kwargs(tmp_path),
            run_id="planning-failure-check",
        )


def test_gen_queries_propagates_realization_failure_and_stops_downstream_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realization failures propagate and prevent assembly and export."""
    monkeypatch.setattr(
        querygen_api,
        "run_realization_stage",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("realization failed")),
    )
    monkeypatch.setattr(
        querygen_api,
        "assemble_query_rows",
        lambda **kwargs: pytest.fail("assemble_query_rows must not run after realization failure"),
    )
    monkeypatch.setattr(
        querygen_api,
        "export_queries",
        lambda **kwargs: pytest.fail("export_queries must not run after realization failure"),
    )
    monkeypatch.setattr(
        querygen_api,
        "export_planning_summary",
        lambda **kwargs: pytest.fail("export_planning_summary must not run after realization failure"),
    )

    with pytest.raises(RuntimeError, match="realization failed"):
        querygen_api.gen_queries(
            **_required_querygen_kwargs(tmp_path),
            run_id="realization-failure-check",
        )


def test_gen_queries_handles_empty_selected_blueprints_after_stage1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When nothing survives stage 1, realization is skipped and empty outputs are assembled/exported."""
    assemble_rows_calls: list[dict[str, object]] = []
    assemble_meta_calls: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        querygen_api,
        "deduplicate_blueprints",
        lambda candidates, near_duplicate_tolerance=0.95: [],
    )
    monkeypatch.setattr(
        querygen_api,
        "run_realization_stage",
        lambda **kwargs: pytest.fail("run_realization_stage must not run when no blueprints remain"),
    )

    def assemble_query_rows(
        *,
        blueprints: list[QueryBlueprint],
        realized_queries: list[RealizedQuery],
        run_id: str,
    ) -> list[SyntheticQueryRow]:
        assemble_rows_calls.append(
            {
                "blueprints": blueprints,
                "realized_queries": realized_queries,
                "run_id": run_id,
            }
        )
        return []

    def assemble_queries_meta(
        *,
        run_id: str,
        n_requested_queries: int,
        n_returned_queries: int,
        model_provider: str,
        planning_model: str,
        realization_model: str,
    ) -> SyntheticQueriesMeta:
        assemble_meta_calls.append(
            {
                "run_id": run_id,
                "n_requested_queries": n_requested_queries,
                "n_returned_queries": n_returned_queries,
                "model_provider": model_provider,
                "planning_model": planning_model,
                "realization_model": realization_model,
            }
        )
        return _make_meta(
            run_id=run_id,
            n_requested_queries=n_requested_queries,
            n_returned_queries=n_returned_queries,
            model_provider=model_provider,
            planning_model=planning_model,
            realization_model=realization_model,
        )

    def export_queries(
        *,
        rows: list[SyntheticQueryRow],
        meta: SyntheticQueriesMeta,
        queries_path: Path,
        meta_path: Path,
    ) -> None:
        export_calls.append(
            {
                "rows": rows,
                "meta": meta,
                "queries_path": queries_path,
                "meta_path": meta_path,
            }
        )

    monkeypatch.setattr(querygen_api, "assemble_query_rows", assemble_query_rows)
    monkeypatch.setattr(querygen_api, "assemble_queries_meta", assemble_queries_meta)
    monkeypatch.setattr(querygen_api, "export_queries", export_queries)

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=3,
        run_id="empty-selected-blueprints-check",
    )

    assert assemble_rows_calls == [
        {
            "blueprints": [],
            "realized_queries": [],
            "run_id": "empty-selected-blueprints-check",
        }
    ]
    assert assemble_meta_calls == [
        {
            "run_id": "empty-selected-blueprints-check",
            "n_requested_queries": 3,
            "n_returned_queries": 0,
            "model_provider": "mistralai",
            "planning_model": "magistral-medium-latest",
            "realization_model": "mistral-medium-latest",
        }
    ]
    assert export_calls == [
        {
            "rows": [],
            "meta": _make_meta(
                run_id="empty-selected-blueprints-check",
                n_requested_queries=3,
                n_returned_queries=0,
            ),
            "queries_path": result.paths.synthetic_queries_csv,
            "meta_path": result.paths.synthetic_queries_meta_json,
        }
    ]


def test_gen_queries_enable_planning_memory_arg_overrides_config_value(
    tmp_path: Path,
) -> None:
    """Explicit enable_planning_memory arg takes precedence over config value."""
    config_path = tmp_path / "querygen.yml"
    config_path.write_text(
        dedent(
            """\
            llm:
              model_provider: mistralai
            enable_planning_memory: true
            """
        ),
        encoding="utf-8",
    )

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        config_path=config_path,
        enable_planning_memory=False,
        run_id="planning-memory-precedence-check",
    )

    assert result.settings.enable_planning_memory is False


def test_gen_queries_passes_near_duplicate_tolerance_to_deduplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit near_duplicate_tolerance is resolved and forwarded to deduplication."""
    seen: dict[str, object] = {}

    def deduplicate_blueprints(
        candidates: list[QueryBlueprint],
        near_duplicate_tolerance: float = 0.95,
    ) -> list[QueryBlueprint]:
        seen["candidate_ids"] = [candidate.candidate_id for candidate in candidates]
        seen["near_duplicate_tolerance"] = near_duplicate_tolerance
        return candidates

    monkeypatch.setattr(querygen_api, "deduplicate_blueprints", deduplicate_blueprints)

    result = querygen_api.gen_queries(
        **_required_querygen_kwargs(tmp_path),
        n_queries=3,
        near_duplicate_tolerance=0.99,
        run_id="near-duplicate-tolerance-check",
    )

    assert result.settings.near_duplicate_tolerance == 0.99
    assert seen == {
        "candidate_ids": ["c001", "c002", "c003"],
        "near_duplicate_tolerance": 0.99,
    }
