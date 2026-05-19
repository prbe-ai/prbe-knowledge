"""Tests for PagerDuty + incident.io SourceSystem and DocType constants."""

from shared.constants import (
    SOURCE_DISPLAY_NAMES,
    SOURCE_HALF_LIFE_DAYS,
    SOURCE_INGESTION_PRIORITY,
    DocType,
    SourceSystem,
)


def test_pagerduty_source_system_value() -> None:
    assert SourceSystem.PAGERDUTY.value == "pagerduty"


def test_incident_io_source_system_value() -> None:
    assert SourceSystem.INCIDENT_IO.value == "incident_io"


def test_incident_doctype_value() -> None:
    assert DocType.INCIDENT.value == "incident"


def test_incident_investigation_doctype_value() -> None:
    assert DocType.INCIDENT_INVESTIGATION.value == "incident.investigation"


def test_new_sources_in_display_names() -> None:
    assert SOURCE_DISPLAY_NAMES[SourceSystem.PAGERDUTY] == "PagerDuty"
    assert SOURCE_DISPLAY_NAMES[SourceSystem.INCIDENT_IO] == "incident.io"


def test_new_sources_in_ingestion_priority() -> None:
    assert SOURCE_INGESTION_PRIORITY[SourceSystem.PAGERDUTY] == 100
    assert SOURCE_INGESTION_PRIORITY[SourceSystem.INCIDENT_IO] == 100


def test_pagerduty_half_life_days() -> None:
    assert SOURCE_HALF_LIFE_DAYS[SourceSystem.PAGERDUTY] == 200.0


def test_incident_io_half_life_days() -> None:
    assert SOURCE_HALF_LIFE_DAYS[SourceSystem.INCIDENT_IO] == 200.0
