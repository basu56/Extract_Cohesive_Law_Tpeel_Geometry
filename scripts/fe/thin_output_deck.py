"""thin_output_deck.py -- copy of a build_tpeel_inp.py deck with field output thinned to what a tslid.fe
run writes (D25 timing protocol).  The original deck is not modified.

    python3 scripts/fe/thin_output_deck.py <deck.inp> <out.inp>

tslid.fe writes per increment: bond-face U2 (opening profile), normal traction and damage per cohesive
element, RP U/UR/RF, energies (src/tslid/fe/tpeel_output.py).  The copy therefore replaces the
*Output, field blocks (default: U, RF at all nodes and S, E, SDEG, STATUS, QUADSCRT on COH_ALL every
increment; S, PEEQ on ADHERENDS every 10th) by
    *Output, field, frequency=1
    *Node Output, nset=IFACE_U / IFACE_L: U
    *Element Output, elset=COH_ALL: S, SDEG
and keeps the history output and node prints unchanged.  Python 3.6 compatible.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().splitlines()
i0 = next(i for i, l in enumerate(lines) if l.lower().startswith("*output, field"))
i1 = next(i for i, l in enumerate(lines) if l.lower().startswith("*output, history"))
assert i0 < i1
thin = ["** field output thinned to the tslid.fe outputs (scripts/fe/thin_output_deck.py, D25)",
        "*Output, field, frequency=1",
        "*Node Output, nset=IFACE_U", "U",
        "*Node Output, nset=IFACE_L", "U",
        "*Element Output, elset=COH_ALL", "S, SDEG"]
open(dst, "w").write("\n".join(lines[:i0] + thin + lines[i1:]) + "\n")
print("removed lines %d-%d of %s:" % (i0 + 1, i1, src))
print("\n".join(lines[i0:i1]))
