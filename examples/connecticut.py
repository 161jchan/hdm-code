"""Run the horizontal operator on a plan ensemble and report what it finds.

    python examples/connecticut.py path/to/small.jsonl.gz

The file is a gzipped JSON-lines ensemble: three header lines, then one plan
per line, with a "districting" list mapping precinct names to district keys.
"""

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from horizontal_diffusion_map import HorizontalDiffusionMap  # noqa: E402


def load_plans(path):
    plans = []
    with gzip.open(path, "rt") as f:
        for _ in range(3):
            f.readline()
        for line in f:
            plan = {}
            for entry in json.loads(line)["districting"]:
                plan.update(entry)
            plans.append(plan)
    return plans


def precinct_groups(plans):
    """Return data[j]: the encoding of every district that contains precinct j."""
    names = sorted(plans[0])
    index = {name: i for i, name in enumerate(names)}

    districts = []
    for plan in plans:
        for key in set(plan.values()):
            row = np.zeros(len(names), dtype=bool)
            row[[index[name] for name, k in plan.items() if k == key]] = True
            districts.append(row)
    districts = np.array(districts)

    return [districts[districts[:, j]] for j in range(len(names))]


def main(path):
    plans = load_plans(path)
    data = precinct_groups(plans)

    start = time.time()
    hdm = HorizontalDiffusionMap(
        data, alpha=0.5, bandwidth=1.0, name="ct_small", num_evecs=10, n_neighbors=10
    )
    elapsed = time.time() - start

    P = hdm.transition_matrix
    n_components, _ = connected_components(hdm.W, directed=False)
    embedding = hdm.embed(n_dimensions=2, t=1)
    distinct = len(np.unique(np.round(embedding, 8), axis=0))

    print(f"plans: {len(plans)}, precincts: {len(data)}, points: {hdm.num_points}")
    print(f"built in {elapsed:.1f}s, density of P: {P.nnz / P.shape[0] ** 2:.2e}")
    print(f"connected components: {n_components}")
    print(f"distinct embedded points: {distinct}")


if __name__ == "__main__":
    main(sys.argv[1])
