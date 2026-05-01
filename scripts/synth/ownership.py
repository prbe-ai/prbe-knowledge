"""OwnershipIndex — services-per-person derived from git commit history.

WorldModel.Service.owners is always () in Plan 2 (Plan 1 left it blank).
The OwnershipIndex is the workaround: we compute service ownership at
scenario-build time by walking commit file paths through manifest ancestry.

Algorithm:
  For each commit in each RepoSignals:
    1. Resolve author_email to a canonical_id using the same email-lowercasing
       rule as canonicalize_people (gh: prefix if noreply, email: otherwise).
    2. For each file in commit.files_touched, find the Manifest in this
       RepoSignals whose manifest.path.parent is the deepest ancestor of the file.
       The manifest.name is the service name.
    3. Record (canonical_id, service_name) pair.
  Aggregate per person: top-3 service names by frequency, alphabetical tie-break.
  Inverse: people_by_service[service_name] = sorted list of canonical_ids.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.synth.extractor.repo import RepoSignals

# Import the noreply resolver from world_model — same package, justified by
# avoiding duplication of the noreply/name-merge logic.
from scripts.synth.world_model import WorldModel, _gh_username_from_noreply


@dataclass(frozen=True)
class OwnershipIndex:
    services_by_person: dict[str, tuple[str, ...]]  # canonical_id → top-3 service names by frequency (manifest.name)
    people_by_service: dict[str, tuple[str, ...]]   # service name (manifest.name) → sorted canonical_ids


def _resolve_canonical_id(email: str) -> str:
    """Map a commit author_email to a canonical_id string.

    Mirrors the two-rule logic in canonicalize_people:
      - GitHub noreply emails → gh:<username>
      - All others → email:<lowercased>
    Does NOT consult the Contributor list (no GH API data available here).
    """
    email_lower = email.lower().strip()
    username = _gh_username_from_noreply(email)
    if username:
        return f"gh:{username}"
    return f"email:{email_lower}"


def _strip_path_root(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Drop leading root components so path parts are meaningful relative parts.

    Removes a leading '/' (POSIX root) or any '<drive>:' component (Windows),
    returning only the meaningful relative segments of the path.
    """
    if not parts:
        return parts
    head = parts[0]
    if head in ("/", "\\") or (len(head) == 2 and head[1] == ":"):
        return parts[1:]
    return parts


def _deepest_manifest_ancestor(
    file_path: str,
    manifests: tuple,
) -> str | None:
    """Return the manifest name whose directory is the deepest ancestor of file_path.

    file_path is a repo-relative string like "services/payments/src/main.py".
    We compare it against each manifest's path.parent, which may be absolute
    or relative. The deepest match (longest meaningful path prefix) wins.

    Matching strategy: strip root components from the manifest directory parts
    via _strip_path_root, then check whether file_path starts with that
    relative prefix. The longest matching prefix (highest matched_depth) wins.

    Example: manifests at "/repo/services/payments" and "/repo/services" with
    file "services/payments/handler.py" -> matched_depth=2 (payments wins).
    """
    file_parts = Path(file_path).parts
    best_name: str | None = None
    best_depth: int = -1

    for m in manifests:
        if not m.name:
            continue

        manifest_dir = m.path.parent
        manifest_dir_parts = manifest_dir.parts

        # Try every suffix length from full length down to 1.
        # The deepest (longest suffix) that matches file_path wins.
        matched_depth = -1
        for suffix_len in range(len(manifest_dir_parts), 0, -1):
            suffix = _strip_path_root(manifest_dir_parts[-suffix_len:])
            # After stripping, an empty suffix means the slice was only root
            # components — skip it.
            if not suffix:
                continue
            if len(suffix) > len(file_parts):
                continue
            if file_parts[: len(suffix)] == suffix:
                matched_depth = suffix_len
                break  # longest matching suffix found

        if matched_depth > best_depth:
            best_depth = matched_depth
            best_name = m.name

    return best_name


def build_ownership_index(
    signals: list[RepoSignals],
    world: WorldModel,
) -> OwnershipIndex:
    """Build OwnershipIndex from commit history.

    Does not modify WorldModel. Safe to call multiple times with the same
    inputs (pure function of signals + world).
    """
    # (canonical_id, service_name) -> count
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for sig in signals:
        for commit in sig.commits:
            canonical_id = _resolve_canonical_id(commit.author_email)
            for file_path in commit.files_touched:
                svc_name = _deepest_manifest_ancestor(file_path, sig.manifests)
                if svc_name:
                    pair_counts[(canonical_id, svc_name)] += 1

    # Aggregate per person: top-3 by frequency, alphabetical tie-break.
    person_service_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for (canonical_id, svc_name), count in pair_counts.items():
        person_service_counts[canonical_id][svc_name] = count

    services_by_person: dict[str, tuple[str, ...]] = {}
    for canonical_id, counts in person_service_counts.items():
        # Sort by (-frequency, name) for deterministic alphabetical tie-break.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        services_by_person[canonical_id] = tuple(name for name, _ in ranked[:3])

    # Inverse: service -> sorted list of people who touched it >= 1 time.
    service_people: dict[str, set[str]] = defaultdict(set)
    for (canonical_id, svc_name) in pair_counts:
        service_people[svc_name].add(canonical_id)

    people_by_service: dict[str, tuple[str, ...]] = {
        svc: tuple(sorted(people))
        for svc, people in service_people.items()
    }

    return OwnershipIndex(
        services_by_person=services_by_person,
        people_by_service=people_by_service,
    )
