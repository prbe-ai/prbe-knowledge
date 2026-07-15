"""Source profiles for incident sources that have no Connector class (yet).

PagerDuty and incident.io documents (DocType.INCIDENT /
DocType.INCIDENT_INVESTIGATION) are written by the incident pipeline rather
than a webhook Connector, so there is no class for @register_connector to
hang their metadata on. Register the profiles directly; when a real
connector lands for either source, move these values onto the class and
delete the explicit registration here.
"""

from __future__ import annotations

from shared.constants import SourceSystem
from shared.source_registry import SourceProfile, register_source

# Incident sources are first-class authored signals: same queue tier as
# Slack / Linear / GitHub webhooks (100, the default — explicit here so the
# intent is obvious), and incident records remain relevant for post-mortems
# and pattern-matching for months: 200d keeps them well above the 120d
# baseline while still decaying relative to very recent incidents.
for _source in (SourceSystem.PAGERDUTY, SourceSystem.INCIDENT_IO):
    register_source(
        SourceProfile(
            source_key=_source.value,
            doc_type_prefix="incident.",
            ingestion_priority=100,
            half_life_days=200.0,
        )
    )
