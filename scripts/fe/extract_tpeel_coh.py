"""extract_tpeel_coh.py -- abaqus python: cohesive-row S and E of every frame of a T-peel ODB.

    abaqus python scripts/fe/extract_tpeel_coh.py <job>.odb [...]

Writes <job>_coh.csv: frame, time, el (COH2D4 label), ip, then every component of S and E (by
label).  Only COH2D4 values are kept (adherend S shares the field).  Units: MPa, mm/mm (constitutive
thickness 1 -> E = separation in mm).  Single precision ODB (~1e-7 relative).
"""
import sys

from odbAccess import openOdb


def main(paths):
    for path in paths:
        odb = openOdb(path, readOnly=True)
        step = odb.steps[list(odb.steps.keys())[-1]]
        out, header = open(path[:-4] + '_coh.csv', 'w'), False
        for k, fr in enumerate(step.frames):
            fo = fr.fieldOutputs
            if 'S' not in fo.keys():
                continue
            S, E = fo['S'], fo['E']
            if not header:
                out.write(','.join(['frame', 'time', 'el', 'ip'] + ['S_' + c for c in S.componentLabels] +
                                   ['E_' + c for c in E.componentLabels]) + '\n')
                header = True
            Ev = dict(((v.elementLabel, v.integrationPoint), v.data) for v in E.values
                      if v.baseElementType == 'COH2D4')
            for v in S.values:
                if v.baseElementType != 'COH2D4':
                    continue
                e = Ev[(v.elementLabel, v.integrationPoint)]
                out.write(','.join(['%d' % k, '%.10g' % fr.frameValue, '%d' % v.elementLabel, '%d' % v.integrationPoint] +
                                   ['%.9g' % x for x in v.data] + ['%.9g' % x for x in e]) + '\n')
        out.close()
        odb.close()


if __name__ == '__main__':
    main(sys.argv[1:])
