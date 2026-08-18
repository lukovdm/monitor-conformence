"""L#□ (L#box) baseline: verbatim copy of https://github.com/JLaumen/l-sharp-square-algorithm

Commit 3fbb416ab0387aa780e126f159dc08999423c1d1, accompanying the CAV paper
"An L# Based Algorithm for Active Learning of Minimal Separating Automata"
(Laumen, Snel, Vaandrager). Upstream spells the □ of the algorithm's name
"square" in its repository and file names; those names are kept as-is so the
copy stays verbatim.

The files in this package are byte-identical to upstream so the baseline can be
re-synced with a plain copy. They import each other by top-level module name
(``from MooreNode import MooreNode``), so this package puts its own directory
on ``sys.path`` before any of them is loaded. Everything ToVer needs to bridge
the two interfaces lives outside this folder: ``ScalarQuerySUL`` in
``tover/core/sul.py`` adapts AALpy's per-symbol ``query`` to the single-output
``query`` upstream expects.
"""

import sys
from pathlib import Path

_UPSTREAM_DIR = str(Path(__file__).parent)
if _UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, _UPSTREAM_DIR)
