# Host environments

Use a repository-local virtual environment named `isac` on every host. Follow [S0 and environment provisioning](../docs/dev/ENVIRONMENT.md) for the core and full profiles, role-specific package installation, runtime probes, manifests and multi-machine handoff. Build the pinned native receiver using the [S1 native runbook](../docs/dev/NATIVE_BASELINE.md). Keep compiled objects, the interpreter and site packages local to their machine. Offline CFRs and experiment manifests are portable by contract.
