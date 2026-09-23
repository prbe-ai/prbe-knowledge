# Entity auto-merge replay (Jev vs gpt-oss)

The eval for `engine/ingest/auto_merge/jev_judge.py`. Re-run it before changing
the Choice instructions, the criteria wording, the value trimming, or the
0.95 / 0.70 bands: each of those is a model change, and
`tests/test_auto_merge_jev_judge.py` pins the instruction text so a change
cannot slip through unnoticed.

Read-only against the kb database. It never merges, suggests or writes.

```bash
export KB_PSQL='<psql command for the kb db, -q -A -t, reading SQL on stdin>'   # see kb.py
python -m scripts.jev_automerge.build_set --customer <id> --out ~/replay/<date> \
    [--logged post_write_done.jsonl]          # uniques only exist in worker logs
TYPESAFE_API_KEY=... python -m scripts.jev_automerge.replay \
    --decisions ~/replay/<date>/decisions.jsonl --out ~/replay/<date>/results.jsonl \
    [--gptoss http://<litellm>/v1]            # with LLM_GATEWAY_KEY; ~$0.0007 a call
python -m scripts.jev_automerge.score --set ~/replay/<date> --results ~/replay/<date>/results.jsonl
```

- **Output directories hold tenant data** (names, emails, ids). Keep them out of
  this repository.
- `build_set` rebuilds each decision's candidates with the analyzer's own
  trigram SQL (as the app role, under RLS) and the same exact kNN. The
  production vector query is a ~20 s seq scan per Document, so the embeddings
  are pulled once and the kNN is computed locally instead.
- `score` exits 1 when the acceptance bar fails. The bar:
  - 0 known-false and 0 unverified Jev auto-merges.
  - At least 12 verified auto-merges (people confirmed by the gate count).
    The analyzer never judges a document's own node, so `build_set` records
    `is_document` and `score` skips those decisions: the 2026-09-23 set leaves
    14 of 82. For non-Person pairs "verified" is mostly the gate's own
    evidence, so it shows agreement, not independent truth.
  - At least 94% same-entity agreement with gpt-oss.

  Results on 2026-09-23 are in `docs/jev-contract.md` ("Entity auto-merge").
