"""build_s3a_elastic.py -- S3a elastic-gate decks with the mesh of a reference T-peel job.

Units: mm, N, MPa, s.  Light; runs anywhere (Python 3.6+, uses only src/tslid/build_tpeel_inp.py).

    python scripts/fe/build_s3a_elastic.py <ref job .json> <outdir> [--n-inc 4] [--delta 1.6]

Calls the unchanged ABAQUS-side builder with every parameter of the reference job (so the mesh is
identical node for node) except --law elastic and --n-inc / --job.  Writes two decks:
    s3a_el_default.inp   ABAQUS default convergence controls
    s3a_el_tight.inp     same deck plus, right after *Static,
                         *Controls, parameters=field, field=global  /  1e-9, 1e-9
                         (R_n and C_n tightened so that ABAQUS's own Newton tolerance does not
                         limit the comparison; the deck is otherwise identical)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src', 'tslid'))
import build_tpeel_inp as B  # noqa: E402

SKIP = {'job', 'outdir', 'law', 'n_inc', 'delta'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref_json')
    ap.add_argument('outdir')
    ap.add_argument('--n-inc', type=int, default=4)
    ap.add_argument('--delta', type=float, default=1.6)
    a = ap.parse_args()
    params = json.load(open(a.ref_json))['params']
    o = B.parse_args(['--job', 's3a_el_default', '--outdir', a.outdir, '--law', 'elastic',
                      '--n-inc', str(a.n_inc), '--delta', str(a.delta)])
    for k, v in params.items():
        if k not in SKIP and hasattr(o, k):
            setattr(o, k, v)
    nodes, els, sets, rps, law, geo = B.build(o)
    path, meta = B.write_inp(o, nodes, els, sets, rps, law, geo)
    print('wrote', path, geo['n_nodes'], 'nodes', geo['n_elems'], 'elements')
    txt = open(path).read()
    i = txt.index('*Static\n')
    j = txt.index('\n', i + len('*Static\n')) + 1
    tight = txt[:j] + '*Controls, parameters=field, field=global\n1e-9, 1e-9\n' + txt[j:]
    tight = tight.replace('T-peel s3a_el_default', 'T-peel s3a_el_tight')
    with open(os.path.join(a.outdir, 's3a_el_tight.inp'), 'w') as f:
        f.write(tight)
    meta['params']['job'] = 's3a_el_tight'
    with open(os.path.join(a.outdir, 's3a_el_tight.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print('wrote', os.path.join(a.outdir, 's3a_el_tight.inp'))


if __name__ == '__main__':
    main()
