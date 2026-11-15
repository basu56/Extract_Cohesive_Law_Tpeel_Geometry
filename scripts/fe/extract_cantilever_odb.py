"""extract_cantilever_odb.py -- abaqus python: nodal U and RF of every frame -> <job>_nodes.csv.

Units: mm, N.  Run:  abaqus python scripts/fe/extract_cantilever_odb.py <job>.odb [...]
Columns: frame, lam (step time = load factor), node, U1, U2, RF1, RF2.
ODB field output is single precision (~1e-7 relative), well below the S1 thresholds.

With --history-node <label> instead, writes <job>_hist_<label>.csv with every history output
of that node (used for UR3 of the T-peel load point RP1).
"""
import sys

from odbAccess import openOdb


def nodes_csv(path):
    odb = openOdb(path, readOnly=True)
    step = odb.steps[list(odb.steps.keys())[-1]]
    out = open(path[:-4] + '_nodes.csv', 'w')
    out.write('frame,lam,node,U1,U2,RF1,RF2\n')
    for k, fr in enumerate(step.frames):
        U = dict((v.nodeLabel, v.data) for v in fr.fieldOutputs['U'].values)
        RF = dict((v.nodeLabel, v.data) for v in fr.fieldOutputs['RF'].values)
        for n in sorted(U):
            u, r = U[n], RF.get(n, (0.0, 0.0))
            out.write('%d,%.9g,%d,%.9g,%.9g,%.9g,%.9g\n' % (k, fr.frameValue, n, u[0], u[1], r[0], r[1]))
    out.close()
    odb.close()


def history_csv(path, label):
    odb = openOdb(path, readOnly=True)
    out = open(path[:-4] + '_hist_%s.csv' % label, 'w')
    for sname, step in odb.steps.items():
        for rname, reg in step.historyRegions.items():
            if not rname.endswith('.' + label):
                continue
            keys = sorted(reg.historyOutputs.keys())
            out.write('step,time,' + ','.join(keys) + '\n')
            cols = [reg.historyOutputs[k].data for k in keys]
            for i in range(len(cols[0])):
                out.write('%s,%.9g,' % (sname, cols[0][i][0]) +
                          ','.join('%.9g' % c[i][1] for c in cols) + '\n')
    out.close()
    odb.close()


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--history-node':
        label = args[1]
        for p in args[2:]:
            history_csv(p, label)
    else:
        for p in args:
            nodes_csv(p)
