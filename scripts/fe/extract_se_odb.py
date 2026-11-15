"""extract_se_odb.py -- abaqus python: per-frame, per-integration-point response of a single COH2D4.

    abaqus python scripts/fe/extract_se_odb.py <job>.odb [...]

Writes <job>_ip.csv with columns: frame, time, ip, then every component of S and E (by label),
SDEG, and the energies ALLSE, ALLDMD, ALLVD, ALLWK at that frame (history, nearest time).
Units: mm, N, MPa, s.  Dependency-free apart from odbAccess.
"""
import sys

from odbAccess import openOdb


def nearest(series, t):
    if not series:
        return float('nan')
    return min(series, key=lambda tv: abs(tv[0] - t))[1]


def main(paths):
    for path in paths:
        odb = openOdb(path, readOnly=True)
        step = odb.steps[list(odb.steps.keys())[-1]]
        H = {}
        for reg in step.historyRegions.values():
            for k in ('ALLSE', 'ALLDMD', 'ALLVD', 'ALLWK'):
                if k in reg.historyOutputs:
                    H[k] = list(reg.historyOutputs[k].data)
        f0 = step.frames[0].fieldOutputs
        ek = 'E' if 'E' in f0.keys() else 'LE'          # NLGEOM writes the strain request as LE
        sl, el = list(f0['S'].componentLabels), list(f0[ek].componentLabels)
        out = open(path[:-4] + '_ip.csv', 'w')
        out.write(','.join(['frame', 'time', 'ip'] + sl + el + ['SDEG', 'ALLSE', 'ALLDMD', 'ALLVD', 'ALLWK']) + '\n')
        for k, fr in enumerate(step.frames):
            fo = fr.fieldOutputs
            S = dict((v.integrationPoint, v.data) for v in fo['S'].values)
            E = dict((v.integrationPoint, v.data) for v in fo[ek].values)
            D = dict((v.integrationPoint, v.data) for v in fo['SDEG'].values) if 'SDEG' in fo.keys() else {}
            en = ['%.10g' % nearest(H.get(n, []), fr.frameValue) for n in ('ALLSE', 'ALLDMD', 'ALLVD', 'ALLWK')]
            for ip in sorted(S):
                row = ['%d' % k, '%.10g' % fr.frameValue, '%d' % ip]
                row += ['%.10g' % x for x in S[ip]] + ['%.10g' % x for x in E[ip]]
                row += ['%.10g' % D.get(ip, float('nan'))] + en
                out.write(','.join(row) + '\n')
        out.close()
        odb.close()


if __name__ == '__main__':
    main(sys.argv[1:])
