import pytest
from services.kg.schema import (
    BugClass, Frontmatter, Signature, Related, Evidence
)
from services.kg.kg_check import check_class, KgCheckError


def _cls(class_id: str, related: Related, body: str = "") -> BugClass:
    return BugClass(
        frontmatter=Frontmatter(
            id=class_id,
            type="bug-class",
            description="x",
            signature=Signature(must_match=["status_code == 200"], embedding_seed="seed text"),
            related=related,
            context_sources=[],
            evidence=Evidence(),
        ),
        body=body,
    )


def test_resolves_when_all_targets_exist():
    universe = {"target-a", "target-b", "self-class"}
    cls = _cls("self-class", Related(analogous_to=["target-a"]), body="See [[target-b]].")
    check_class(cls, universe=universe)  # no exception


def test_fails_on_missing_class_in_related():
    universe = {"self-class"}
    cls = _cls("self-class", Related(analogous_to=["does-not-exist"]))
    with pytest.raises(KgCheckError, match="does-not-exist"):
        check_class(cls, universe=universe)


def test_fails_on_missing_class_in_body():
    universe = {"self-class"}
    cls = _cls("self-class", Related(), body="See [[ghost-class]].")
    with pytest.raises(KgCheckError, match="ghost-class"):
        check_class(cls, universe=universe)


def test_source_links_skipped_by_default():
    universe = {"self-class"}
    cls = _cls("self-class", Related(), body="In [[src/x.ts#fn]].")
    check_class(cls, universe=universe)  # no exception (source refs not validated)


def test_aggregates_multiple_missing():
    universe = {"self-class"}
    cls = _cls(
        "self-class",
        Related(analogous_to=["miss-a"], overlaps_with=["miss-b"]),
        body="See [[miss-c]] and [[miss-d]].",
    )
    with pytest.raises(KgCheckError) as exc:
        check_class(cls, universe=universe)
    msg = str(exc.value)
    for missing in ("miss-a", "miss-b", "miss-c", "miss-d"):
        assert missing in msg


def test_self_reference_allowed():
    universe = {"self-class"}
    # A class referencing itself is fine — it's already in scope.
    cls = _cls("self-class", Related(analogous_to=["self-class"]), body="See [[self-class]].")
    check_class(cls, universe=universe)  # no exception
