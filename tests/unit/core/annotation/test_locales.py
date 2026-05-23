"""Tests for locale catalogs and locale-aware task-settings construction."""

import argilla as rg
import pytest

from pragmata.core.annotation.argilla_task_definitions import build_task_settings
from pragmata.core.annotation.locales.registry import CATALOGS, get_catalog
from pragmata.core.schemas.annotation_task import Locale, Task

_NON_EN_LOCALES = sorted(loc for loc in CATALOGS if loc != "en")
_ALL_LOCALES = sorted(CATALOGS)


class TestCatalogCompleteness:
    """Every non-default locale must define the exact key set the default (EN) provides."""

    @pytest.mark.parametrize("locale", _NON_EN_LOCALES)
    def test_locale_keys_match_english(self, locale: Locale) -> None:
        en_keys = set(get_catalog("en").keys())
        other_keys = set(get_catalog(locale).keys())
        missing = en_keys - other_keys
        extra = other_keys - en_keys
        assert not missing, f"{locale} catalog missing keys: {missing}"
        assert not extra, f"{locale} catalog has unexpected keys: {extra}"

    @pytest.mark.parametrize("locale", _ALL_LOCALES)
    def test_no_empty_strings(self, locale: Locale) -> None:
        catalog = get_catalog(locale)
        empty = [k for k, v in catalog.items() if not v.strip()]
        assert not empty, f"{locale} catalog has empty strings at keys: {empty}"


class TestLocaleAwareSettings:
    """build_task_settings() returns locale-appropriate titles while preserving structure."""

    def test_default_locale_is_english(self) -> None:
        # Cached separately from explicit "en" call, but should produce identical content.
        default_settings = build_task_settings()
        en_settings = build_task_settings("en")
        for task in Task:
            assert _titles(default_settings[task]) == _titles(en_settings[task])

    def test_de_titles_differ_from_en(self) -> None:
        en = build_task_settings("en")
        de = build_task_settings("de")
        # At least some titles must differ — otherwise translations aren't wired through.
        for task in Task:
            en_titles = _titles(en[task])
            de_titles = _titles(de[task])
            assert en_titles != de_titles, f"{task.value}: DE titles identical to EN — wiring broken"

    @pytest.mark.parametrize("locale", _ALL_LOCALES)
    def test_field_and_question_names_stable_across_locales(self, locale: Locale) -> None:
        """Identities (name=) must not change with locale — exports depend on this."""
        en = build_task_settings("en")
        other = build_task_settings(locale)
        for task in Task:
            assert [f.name for f in en[task].fields] == [f.name for f in other[task].fields]
            assert [q.name for q in en[task].questions] == [q.name for q in other[task].questions]

    @pytest.mark.parametrize("locale", _ALL_LOCALES)
    def test_label_values_stable_across_locales(self, locale: Locale) -> None:
        """Label values (e.g. 'yes'/'no') must not change with locale — export parsing depends on this."""
        en = build_task_settings("en")
        other = build_task_settings(locale)
        for task in Task:
            for en_q, other_q in zip(en[task].questions, other[task].questions, strict=True):
                if isinstance(en_q, rg.LabelQuestion):
                    assert isinstance(other_q, rg.LabelQuestion)
                    assert [_label_value(label) for label in en_q.labels] == [
                        _label_value(label) for label in other_q.labels
                    ]

    def test_label_displays_differ_in_de(self) -> None:
        """Label display text (option text shown in the UI) must change with locale."""
        en = build_task_settings("en")
        de = build_task_settings("de")
        seen_any_diff = False
        for task in Task:
            for en_q, de_q in zip(en[task].questions, de[task].questions, strict=True):
                if not isinstance(en_q, rg.LabelQuestion):
                    continue
                assert isinstance(de_q, rg.LabelQuestion)
                en_displays = _option_texts(en_q)
                de_displays = _option_texts(de_q)
                if en_displays != de_displays:
                    seen_any_diff = True
        assert seen_any_diff, (
            "DE label displays identical to EN across all questions — label translation not wired through"
        )

    def test_label_yes_no_displays_in_de(self) -> None:
        """Spot-check: the 'yes'/'no' label values render as 'Ja'/'Nein' in DE."""
        de = build_task_settings("de")
        q = next(q for q in de[Task.GROUNDING].questions if q.name == "support_present")
        assert isinstance(q, rg.LabelQuestion)
        assert _option_map(q) == {"yes": "Ja", "no": "Nein"}

    def test_discard_reason_labels_translated_in_de(self) -> None:
        """Spot-check: discard reasons are translated, while their VALUES stay as DiscardReason.*.value."""
        de = build_task_settings("de")
        q = next(q for q in de[Task.GROUNDING].questions if q.name == "discard_reason")
        assert isinstance(q, rg.LabelQuestion)
        opts = _option_map(q)
        assert set(opts.keys()) == {"invalid_or_unrealistic", "unclear", "outside_reviewer_expertise"}
        # All displays should be non-English (cheap heuristic: differ from the EN spelling).
        en = build_task_settings("en")
        en_q = next(q for q in en[Task.GROUNDING].questions if q.name == "discard_reason")
        assert isinstance(en_q, rg.LabelQuestion)
        assert _option_map(q) != _option_map(en_q)


def _discard_field(locale: Locale) -> rg.CustomField:
    settings = build_task_settings(locale)
    return next(f for f in settings[Task.GROUNDING].fields if f.name == "discard_flow")


class TestDiscardWidgetI18n:
    """Verify the discard widget template embeds locale-aware strings."""

    @pytest.mark.parametrize("locale", _ALL_LOCALES)
    def test_widget_carries_all_locales_in_payload(self, locale: Locale) -> None:
        """Every locale's widget strings ship inside the rendered template."""
        widget = _discard_field(locale)
        for other_locale in _ALL_LOCALES:
            sample = get_catalog(other_locale)[(Task.GROUNDING, "widget", "discard.button_label")]
            assert sample in widget.template, (
                f"Rendered widget (dataset locale={locale}) missing {other_locale} button label {sample!r}"
            )

    def test_widget_default_locale_matches_dataset_locale(self) -> None:
        """The widget's DEFAULT_LOCALE constant points at the dataset's creation locale."""
        assert 'var DEFAULT_LOCALE = "en"' in _discard_field("en").template
        assert 'var DEFAULT_LOCALE = "de"' in _discard_field("de").template

    def test_widget_dataset_locale_listed_first_in_supported_locales(self) -> None:
        """SUPPORTED_LOCALES is ordered with the dataset's locale first, for aria-label probing."""
        assert 'var SUPPORTED_LOCALES = ["de"' in _discard_field("de").template

    def test_widget_aria_label_titles_translated_per_locale(self) -> None:
        """Widget bundles every locale's question titles for aria-label probing."""
        de_template = _discard_field("de").template
        # Both EN and DE title strings present so the widget can find the
        # hidden cards regardless of which chrome locale is currently active.
        for locale in _ALL_LOCALES:
            title = get_catalog(locale)[(Task.GROUNDING, "question", "discard_reason")]
            assert title in de_template, (
                f"Rendered widget (dataset locale=de) missing {locale} discard_reason title {title!r}"
            )


def _titles(settings: rg.Settings) -> dict[str, str]:
    by_name = {f.name: f.title for f in settings.fields}
    by_name.update({q.name: q.title for q in settings.questions})
    by_name["__guidelines__"] = settings.guidelines or ""
    return by_name


def _label_value(label: object) -> str:
    # Argilla's LabelQuestion.labels property exposes just the option *keys*
    # (machine values). Useful when checking value-stability across locales.
    return label if isinstance(label, str) else getattr(label, "value", str(label))


def _option_texts(question: rg.LabelQuestion) -> list[str]:
    """Ordered list of UI display strings for a LabelQuestion's options.

    Reads the underlying ``_model.settings.options`` because the public
    ``.labels`` getter is lossy — it returns only the option keys (values)
    and drops the per-option display ``text``.
    """
    return [opt["text"] for opt in question._model.settings.options]


def _option_map(question: rg.LabelQuestion) -> dict[str, str]:
    """Value-to-display-text map for a LabelQuestion's options."""
    return {opt["value"]: opt["text"] for opt in question._model.settings.options}
