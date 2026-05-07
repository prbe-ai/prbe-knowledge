"""prbe-knowledge generic cron host.

This package exists primarily as a build-context anchor for
`fly.cron.toml` + `services/cron/Dockerfile`. The actual scheduled
commands live in their own modules (e.g. `services.synthesis.nightly_trigger`,
`scripts.leiden_one_shot`); this image just packages all of them
together so a single fly app can host every nightly job in the repo.
"""
