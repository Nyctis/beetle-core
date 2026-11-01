"""The buyer's side: `beetle`. One daemon holds the wallet, the policy and the jobs;
the command line, the Python SDK and the MCP server all talk to it. See client/daemon.py."""

import node  # noqa: F401  (import paths)
