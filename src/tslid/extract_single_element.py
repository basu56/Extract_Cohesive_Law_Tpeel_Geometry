"""
extract_single_element.py -- abaqus python; per-frame response of a single_element.py run.

    abaqus python extract_single_element.py <job>.odb [...]

Writes <job>_se.csv: time, d (E22 mean over integration points), traction (S22 mean),
SDEG (mean), RF2_top (sum over top nodes, N), ALLSE, ALLDMD, ALLVD, ALLWK.
Dependency-free apart from odbAccess.
"""
import os
import sys

from odbAccess import openOdb


def hist(step, name):
    for reg in step.historyRegions.values():
        if name in reg.historyOutputs:
            return list(reg.historyOutputs[name].data)
    return []


def at(series, t):
    """Value of a (time, value) history at the sample nearest to t."""
    if not series:
        return float('nan')
    return min(series, key=lambda tv: abs(tv[0] - t))[1]


def mean(vals):
    return sum(vals) / float(len(vals)) if vals else float('nan')


def main(paths):
    for path in paths:
        odb = openOdb(path, readOnly=True)
        step = odb.steps.values()[-1]
        H = dict((k, hist(step, k)) for k in ('ALLSE', 'ALLDMD', 'ALLVD', 'ALLWK'))
        rows = []
        for fr in step.frames:
            fo = fr.fieldOutputs
            iS = list(fo['S'].componentLabels).index('S22')
            iE = list(fo['E'].componentLabels).index('E22')
            s22 = mean([v.data[iS] for v in fo['S'].values])
            e22 = mean([v.data[iE] for v in fo['E'].values])
            sd = mean([v.data for v in fo['SDEG'].values])
            rf = 0.0
            for v in fo['RF'].values:
                if v.nodeLabel in (3, 4):
                    rf += v.data[1]
            t = fr.frameValue
            rows.append([t, e22, s22, sd, rf] + [at(H[k], t) for k in
                                                ('ALLSE', 'ALLDMD', 'ALLVD', 'ALLWK')])
        out = os.path.splitext(path)[0] + '_se.csv'
        with open(out, 'w') as f:
            f.write('time,d,traction,SDEG,RF2_top,ALLSE,ALLDMD,ALLVD,ALLWK\n')
            for r in rows:
                f.write(','.join('%.10g' % v for v in r) + '\n')
        odb.close()
        print('wrote %s (%d frames)' % (out, len(rows)))


if __name__ == '__main__':
    main(sys.argv[1:])
