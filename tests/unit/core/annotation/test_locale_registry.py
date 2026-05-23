"""Unit tests for register_catalog_dir (user-provided locale catalogs)."""

import shutil
from pathlib import Path

import pytest

from pragmata.core.annotation.locales import registry
from pragmata.core.schemas.annotation_task import Task

BUNDLED_EN = Path(registry.__file__).parent / "en.yaml"


@pytest.fixture
def isolated_catalogs():
    """Restore CATALOGS after each test so registrations don't leak."""
    snapshot = dict(registry.CATALOGS)
    yield
    registry.CATALOGS.clear()
    registry.CATALOGS.update(snapshot)


class TestRegisterCatalogDir:
    def test_adds_new_locale(self, tmp_path, isolated_catalogs):
        shutil.copy(BUNDLED_EN, tmp_path / "xx.yaml")
        assert "xx" not in registry.CATALOGS

        registry.register_catalog_dir(tmp_path)

        assert registry.get_catalog("xx") == registry.get_catalog("en")

    def test_user_overrides_bundled_on_stem_collision(self, tmp_path, isolated_catalogs):
        original = registry.get_catalog("en")[(Task.RETRIEVAL, "field", "query")]
        text = BUNDLED_EN.read_text(encoding="utf-8").replace("query: Query", "query: USER QUERY", 1)
        (tmp_path / "en.yaml").write_text(text, encoding="utf-8")

        registry.register_catalog_dir(tmp_path)

        assert registry.get_catalog("en")[(Task.RETRIEVAL, "field", "query")] == "USER QUERY"
        assert original == "Query"

    def test_idempotent(self, tmp_path, isolated_catalogs):
        shutil.copy(BUNDLED_EN, tmp_path / "yy.yaml")
        registry.register_catalog_dir(tmp_path)
        first = dict(registry.CATALOGS)

        registry.register_catalog_dir(tmp_path)

        assert registry.CATALOGS == first

    def test_empty_dir_noop(self, tmp_path, isolated_catalogs):
        before = dict(registry.CATALOGS)

        registry.register_catalog_dir(tmp_path)

        assert registry.CATALOGS == before
