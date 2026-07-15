"""Tests for PagerDuty + incident.io SourceSystem and DocType constants."""

import services.ingestion.handlers  # noqa: F401  (registers source profiles)
from shared.constants import (
    SOURCE_DISPLAY_NAMES,
    DocType,
    SourceSystem,
)
from shared.source_registry import get_source_profile


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
    assert get_source_profile(SourceSystem.PAGERDUTY.value).ingestion_priority == 100
    assert get_source_profile(SourceSystem.INCIDENT_IO.value).ingestion_priority == 100


def test_pagerduty_half_life_days() -> None:
    assert get_source_profile(SourceSystem.PAGERDUTY.value).half_life_days == 200.0


def test_incident_io_half_life_days() -> None:
    assert get_source_profile(SourceSystem.INCIDENT_IO.value).half_life_days == 200.0
