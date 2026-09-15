"""Does the swap lose a row when ingestion never stops? ATTENDED HARNESS.

    docker exec pg psql -U postgres -c "CREATE DATABASE fencetest OWNER app"
    psql -d fencetest -f scripts/check_swap_under_load.sql
    DATABASE_URL=postgresql://app:app@127.0.0.1:5434/fencetest \
        python scripts/check_swap_under_load.py

Point it at a THROWAWAY database. It creates, swaps and renames tables.

This is the evidence for the fence in `_swap`. Run it against a database shaped
like production -- non-superuser `app` owning `chunks`, FORCE ROW LEVEL
SECURITY, hash-derived `chunk_id`s, the two BEFORE INSERT triggers and a
GENERATED column -- which `check_swap_under_load.sql` builds.

Seeds a prod-shaped `chunks` (FORCE RLS, owner `app`, hash-derived chunk_ids,
two BEFORE INSERT triggers, a GENERATED column), runs a writer that keeps
committing and deleting rows throughout, runs the full conversion, and then
asks the only question that matters: is every row the writer committed and did
not delete present in the table that is now `chunks`?
"""
import asyncio
import hashlib
import json
import os
import random
import sys
import time

import asyncpg

DSN = os.environ["DATABASE_URL"]
TENANTS = ["alpha-co", "beta_co", "gamma-3"]
SEED_PER_TENANT = 4000

committed: dict[str, set[str]] = {t: set() for t in TENANTS}
deleted: dict[str, set[str]] = {t: set() for t in TENANTS}
stop = False
writer_errors: list[str] = []
writer_inserts = 0


def cid(doc: str, body: str) -> str:
    return f"{doc}:{hashlib.sha1(body.encode()).hexdigest()[:16]}"


async def seed(conn):
    for t in TENANTS:
        await conn.execute("SELECT set_config('app.current_customer_id',$1,false)", t)
        for d in range(40):
            await conn.execute(
                "INSERT INTO documents (doc_id, customer_id, version, title, metadata)"
                " VALUES ($1,$2,1,$3,$4::jsonb) ON CONFLICT DO NOTHING",
                f"doc-{d}", t, f"Title {d}", json.dumps({"project_id": f"p{d % 5}"}))
        rows = []
        for i in range(SEED_PER_TENANT):
            doc = f"doc-{i % 40}"
            body = f"seed {t} {i} {random.random()}"
            ck = cid(doc, body)
            rows.append((ck, doc, t, i, body, hashlib.sha1(body.encode()).hexdigest()))
            committed[t].add(ck)
        await conn.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, customer_id, chunk_index,"
            " content, content_hash) VALUES ($1,$2,$3,$4,$5,$6)"
            " ON CONFLICT DO NOTHING", rows)


async def writer(n: int):
    """One live ingestion client. Reconnects, because the swap is a rename."""
    global writer_inserts
    conn = await asyncpg.connect(DSN)
    try:
        while not stop:
            t = random.choice(TENANTS)
            doc = f"doc-{random.randrange(40)}"
            body = f"live {n} {time.time()} {random.random()}"
            ck = cid(doc, body)
            try:
                async with conn.transaction():
                    await conn.execute(
                        "SELECT set_config('app.current_customer_id',$1,true)", t)
                    await conn.execute(
                        "INSERT INTO chunks (chunk_id, doc_id, customer_id,"
                        " chunk_index, content, content_hash)"
                        " VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING",
                        ck, doc, t, 0, body,
                        hashlib.sha1(body.encode()).hexdigest())
                committed[t].add(ck)
                writer_inserts += 1
                # Delete something occasionally, so the reconcile pass matters.
                if random.random() < 0.12 and committed[t] - deleted[t]:
                    victim = random.choice(list(committed[t] - deleted[t]))
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT set_config('app.current_customer_id',$1,true)", t)
                        await conn.execute(
                            "DELETE FROM chunks WHERE customer_id=$1 AND chunk_id=$2",
                            t, victim)
                    deleted[t].add(victim)
            except Exception as exc:
                writer_errors.append(f"{type(exc).__name__}: {exc}")
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.004)
    finally:
        await conn.close()


async def main():
    global stop
    conn = await asyncpg.connect(DSN)
    await seed(conn)
    print(f"seeded {SEED_PER_TENANT * len(TENANTS):,} rows", flush=True)
    await conn.close()

    writers = [asyncio.create_task(writer(i)) for i in range(4)]
    await asyncio.sleep(2)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "scripts.convert_chunks_to_partitioned",
        "--run", "--i-have-stopped-ingestion",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ},
    )
    out, _ = await proc.communicate()
    print(out.decode()[-3000:], flush=True)
    print(f"convert rc={proc.returncode}", flush=True)

    stop = True
    await asyncio.gather(*writers)
    print(f"writer inserts={writer_inserts:,} errors={len(writer_errors)}", flush=True)
    for e in writer_errors[:5]:
        print(f"  writer error: {e}", flush=True)

    conn = await asyncpg.connect(DSN)
    bad = 0
    for t in TENANTS:
        expect = committed[t] - deleted[t]
        await conn.execute("SELECT set_config('app.current_customer_id',$1,false)", t)
        got = {r["chunk_id"] for r in await conn.fetch(
            "SELECT chunk_id FROM chunks WHERE customer_id=$1", t)}
        missing = expect - got
        extra = got - expect
        print(f"{t}: expect={len(expect):,} got={len(got):,} "
              f"MISSING={len(missing):,} EXTRA={len(extra):,}", flush=True)
        bad += len(missing) + len(extra)

    kind = await conn.fetchval(
        "SELECT relkind FROM pg_class WHERE oid='chunks'::regclass")
    trg = [r["tgname"] for r in await conn.fetch(
        "SELECT tgname FROM pg_trigger WHERE tgrelid='chunks'::regclass"
        " AND NOT tgisinternal ORDER BY 1")]
    kind = kind.decode() if isinstance(kind, bytes) else kind
    print(f"chunks relkind={kind!r}  triggers={trg}", flush=True)

    # Do the triggers still fire after the swap?
    await conn.execute("SELECT set_config('app.current_customer_id',$1,false)",
                       "alpha-co")
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, customer_id, chunk_index,"
        " content, content_hash) VALUES ('post-swap:xyz','doc-3','alpha-co',"
        "0,'after','hhh')")
    got = await conn.fetchrow(
        "SELECT project_id, title FROM chunks WHERE chunk_id='post-swap:xyz'")
    print(f"post-swap trigger fill: project_id={got['project_id']!r} "
          f"title={got['title']!r}", flush=True)
    trig_ok = got["project_id"] == "p3" and got["title"] == "Title 3"
    await conn.close()

    print("RESULT:", "PASS" if (bad == 0 and kind == "p" and trig_ok) else "FAIL",
          flush=True)
    return 0 if (bad == 0 and kind == "p" and trig_ok) else 1

sys.exit(asyncio.run(main()))
