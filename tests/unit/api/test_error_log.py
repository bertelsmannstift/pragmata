"""Unit tests for the _error_log context manager."""

import logging

import pytest

from pragmata.api._error_log import error_log


@pytest.fixture()
def log_dir(tmp_path):
    return tmp_path


def _log_file(base):
    return base / "annotation" / "errors.log"


class TestErrorLog:
    def test_error_written_to_file(self, log_dir):
        logger = logging.getLogger("pragmata.test")
        with error_log(log_dir):
            logger.error("something broke")

        lines = _log_file(log_dir).read_text().strip().splitlines()
        assert len(lines) == 1
        assert "something broke" in lines[0]

    def test_critical_written_to_file(self, log_dir):
        logger = logging.getLogger("pragmata.test")
        with error_log(log_dir):
            logger.critical("fatal failure")

        assert "fatal failure" in _log_file(log_dir).read_text()

    def test_info_and_warning_excluded(self, log_dir):
        logger = logging.getLogger("pragmata.test")
        with error_log(log_dir):
            logger.info("info msg")
            logger.warning("warning msg")

        assert not _log_file(log_dir).exists()

    def test_no_file_when_no_errors(self, log_dir):
        logger = logging.getLogger("pragmata.test")
        with error_log(log_dir):
            logger.info("all good")

        assert not _log_file(log_dir).exists()

    def test_handler_removed_after_exit(self, log_dir):
        root = logging.getLogger("pragmata")
        before = len(root.handlers)
        with error_log(log_dir):
            pass
        assert len(root.handlers) == before

    def test_handler_removed_on_exception(self, log_dir):
        root = logging.getLogger("pragmata")
        before = len(root.handlers)
        with pytest.raises(ValueError, match="boom"):
            with error_log(log_dir):
                raise ValueError("boom")
        assert len(root.handlers) == before

    def test_multiple_calls_append(self, log_dir):
        logger = logging.getLogger("pragmata.test")
        with error_log(log_dir):
            logger.error("first")
        with error_log(log_dir):
            logger.error("second")

        lines = _log_file(log_dir).read_text().strip().splitlines()
        assert len(lines) == 2
        assert "first" in lines[0]
        assert "second" in lines[1]

    def test_format_includes_timestamp_and_logger_name(self, log_dir):
        logger = logging.getLogger("pragmata.core.annotation.setup")
        with error_log(log_dir):
            logger.error("format check")

        line = _log_file(log_dir).read_text().strip()
        assert "pragmata.core.annotation.setup" in line
        assert "ERROR" in line
        assert "format check" in line
