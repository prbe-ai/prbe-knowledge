"""Tests for PagerDuty + incident.io SourceSystem and DocType constants."""

import kb.handlers  # noqa: F401  (registers source profiles)
from engine.shared.constants import (
    PRIORITY_LIVE_INTEGRATION,
    SOURCE_DISPLAY_NAMES,
    DocType,
    SourceSystem,
)
from engine.shared.source_registry import get_source_profile


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
    assert (
        get_source_profile(SourceSystem.PAGERDUTY.value).ingestion_priority
        == PRIORITY_LIVE_INTEGRATION
    )
    assert (
        get_source_profile(SourceSystem.INCIDENT_IO.value).ingestion_priority
        == PRIORITY_LIVE_INTEGRATION
    )


def test_pagerduty_half_life_days() -> None:
    assert get_source_profile(SourceSystem.PAGERDUTY.value).half_life_days == 200.0


def test_incident_io_half_life_days() -> None:
    assert get_source_profile(SourceSystem.INCIDENT_IO.value).half_life_days == 200.0
