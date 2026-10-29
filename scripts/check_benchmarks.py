"""Recompute the published controlled energy and medians from CSV traces."""
from pathlib import Path
import csv
import json
import math
import statistics


def check(root):
    root = Path(root).resolve()
    report = json.loads((root / 'pnnx-controlled.json').read_text())
    for run in report['runs']:
        path = (root / run['power_trace']).resolve()
        if root not in path.parents:
            raise ValueError('Power trace escapes benchmark directory')
        with path.open(newline='', encoding='utf8') as stream:
            points = [(float(row['time_since_first_crop_s']), float(row['system_rail_w']))
                      for row in csv.DictReader(stream)]
        if len(points) < 2 or any(not math.isfinite(t) or not math.isfinite(w) or w < 0 for t, w in points):
            raise ValueError('Invalid power samples')
        if abs(points[0][0]) > 1e-9 or abs(points[-1][0] - run['duration_s']) > 1e-8:
            raise ValueError('Power trace does not cover the measured interval')
        if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise ValueError('Non-monotonic power trace')
        joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:]))
        for actual, expected in [(joules / run['images'], run['joules_per_image']),
                                 (joules / run['duration_s'], run['mean_system_w']),
                                 (run['images'] / run['duration_s'], run['throughput_images_s'])]:
            if abs(actual - expected) > 1e-8:
                raise ValueError('Published measurement does not match its trace')
    for profile, aggregate in report['aggregate'].items():
        runs = [run for run in report['runs'] if run['profile'] == profile]
        if len(runs) != aggregate['repetitions']:
            raise ValueError('Repetition count mismatch')
        for field in ('throughput_images_s', 'mean_system_w', 'joules_per_image'):
            values = [run[field] for run in runs]
            for key, actual in [('median', statistics.median(values)), ('min', min(values)), ('max', max(values))]:
                if abs(actual - aggregate[field][key]) > 1e-10:
                    raise ValueError('Published aggregate mismatch')
    return len(report['runs'])


if __name__ == '__main__':
    print('BENCHMARK_TRACES_VERIFIED', check(Path(__file__).resolve().parents[1] / 'benchmarks'))
