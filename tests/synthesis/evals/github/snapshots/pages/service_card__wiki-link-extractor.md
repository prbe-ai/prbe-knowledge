---
title: "Wiki link extractor"
slug: wiki-link-extractor
wiki_type: service_card
last_updated: 2026-04-23T11:14:08Z
owners: [person:richard]
related: [feature:wiki-bootstrap]
---

# Wiki link extractor

Pure-parser module that pulls `[[type:slug]]` references out of wiki
page bodies (markdown) and YAML frontmatter, then writes them to
`wiki_links` in a delete-then-insert pattern. No LLM calls per write —
the writer hook runs the parser inline.

Lives in `services/synthesis/wiki_links.py`.

## Recent changes

- 2026-04-23 — [[person:richard]] shipped the initial extractor +
  persister in [PR #45](https://github.com/prbe-ai/prbe-knowledge/pull/45),
  with [[person:maison|reviewer]] and [[person:janedoe|reviewer]] both
  signing off after the regex was tightened to not span line breaks
  and a nested-bracket test was added.

## Sources

- [PR #45: wiki: ship typed-link extractor + persister](https://github.com/prbe-ai/prbe-knowledge/pull/45)
