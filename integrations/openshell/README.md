# OpenShell: confining the buyer's agent

**Not run.** All that is here is a policy written from the documentation.

What `policy.yaml` intends: an agent handling private data can reach only the local Guardrails gate and cannot send anywhere else, including Infinite, the public site ScienceClaw uses by default.

Why it was not run: OpenShell's CLI and gateway can only be installed with the `curl | sh` installer or GitHub release packages, which were not used this time. `openshell` on PyPI (0.1.6, Apache-2.0) is only an SDK of 150,000 bytes, and the 0.0.116 wheel is only 110,000 bytes. Neither contains the CLI or the gateway, and the SDK has to connect to a gateway.

This policy was only read against the schema in `docs/how-it-works/policies/`. OpenShell has never parsed it. To verify before going live:

- Whether the `host.openshell.internal` rule can reach the gate. The documentation says it can reach host services, but the gate binds to `127.0.0.1`, which may be out of reach under the Docker driver. The bind address would have to change, and a gate with no token should not bind to the outside.
- Binary paths. The process that opens the connection is the interpreter, not `scienceclaw-post`, so the paths must be changed to match the real image.
- OpenShell has its own telemetry. Turn it off with `OPENSHELL_TELEMETRY_ENABLED=false`.

It can only go on the buyer side. The policy is controlled by the machine's owner, and on the seller side the adversary is the machine's owner.
