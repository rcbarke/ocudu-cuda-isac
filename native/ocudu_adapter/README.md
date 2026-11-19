# Native replay and observation adapter

First implementation is assigned to S2–S3 in [DEVELOPMENT.md](../../docs/dev/DEVELOPMENT.md). Build a thin native component against the pinned OCUDU grid interfaces. Reuse the legal stimulus, scheduling and production receiver. Hand off a completed grid and a bounded owned pilot snapshot; distinguish producer completion, consumer completion and buffer release. This directory will contain only the adapter and its build wiring. The initial executable Python fixture in `src/` supplies small input/output reference vectors for native tests.
