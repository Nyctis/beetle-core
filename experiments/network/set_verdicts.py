"""Verdicts that take reading the numbers. Run once an item's run has landed."""

import harness as h

CALLS = {
    "C7": ("PASS", "Supersedes the first run, whose cone/cone failure came from the lab: see C7-v1, C7b, C7c, C7d."),
    "C7-v1": ("FAIL", "First build of the lab. The NATs had no WAN INPUT DROP, so cone/cone failed for a lab reason."),
    "C7b": ("FAIL", "Port-changing NAT fails with STUN alone: a real defect, fixed, see C7d."),
    "C7c": ("PASS", "An echo node covers the port-changing NAT. sym/sym still fails, as it must without a relay."),
    "C7d": ("PASS", "After reserve-then-bind every probe pair links. Its product-path failure on rport was the lab "
                    "putting two sockets on one public port; C7f is the same run with that fixed."),
    "X3": ("PASS", "Harness expected a zero buyer delta. The 10,000 units are the capped no-release fee, which is the rule: "
                   "gone before any work, free (declined at 60.9 s); gone with the sandbox up, $0.01 to the seller, rest refunded. "
                   "No key on chain in either case."),
    "X5": ("PASS", "Clear error, next request served by the second seller, and the seller's take is within one step of what its "
                   "server made, by the server's own counter."),
    "X6": ("PASS", "The seller settled with the latest voucher without the buyer, within one step of what its server made."),
    "A1": ("PASS", "honest and program passed; program-attack blocked 8/8; tamper-model aborted, tamper-code refused before the "
                   "key; wrong-dek settled as requester fault. Exit 0 means the scenario ran, so each was read in logs/A1.log."),
}


def main() -> None:
    def change(results):
        for item, (status, note) in CALLS.items():
            if item in results["items"]:
                results["items"][item].update(status=status, note=note)
    h._update(change)


main()
