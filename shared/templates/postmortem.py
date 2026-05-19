"""Default postmortem template used when a customer has no override.

The template uses ``{{slot_name}}`` placeholders that the agent's
PostmortemDraft fields fill at render time. Unknown placeholders render
empty. Missing fields render empty. The slot vocabulary is hardcoded
in v1: summary, impact, timeline, root_cause, contributing_factors,
what_went_well, what_went_wrong, action_items.
"""
from __future__ import annotations

DEFAULT_POSTMORTEM_TEMPLATE: str = """\
# Postmortem: {{title}}

**Incident:** {{incident_link}}
**Investigation:** {{investigation_link}}
**Triggered:** {{triggered_at}}
**Resolved:** {{resolved_at}}
**Severity:** {{severity}}

## Summary

{{summary}}

## Impact

- **Users affected:** {{impact.users_affected}}
- **Duration:** {{impact.duration_minutes}} minutes
- **Services:** {{impact.services}}
- **Severity:** {{impact.severity}}

## Timeline

{{timeline}}

## Root Cause

{{root_cause}}

## Contributing Factors

{{contributing_factors}}

## What Went Well

{{what_went_well}}

## What Went Wrong

{{what_went_wrong}}

## Action Items

{{action_items}}
"""
