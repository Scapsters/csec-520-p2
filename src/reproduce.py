"""One-command experiment: a baseline before implementation, both distances after."""
import argparse
import copy
import json
from pathlib import Path
import yaml
from .data import load_data
from .cluster import run_experiment


def main(config_path='config.yaml', output_dir=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_dir or cfg['output']['dir'])
    out.mkdir(parents=True, exist_ok=True)
    # An interrupted run must not leave an old aggregate that looks successful.
    summary_path = out / 'metrics.json'
    summary_path.unlink(missing_ok=True)
    data = load_data(cfg)
    impl = cfg['kmeans'].get('implementation', 'sklearn')
    if impl == 'sklearn' and cfg['kmeans'].get('distance', 'euclidean') != 'euclidean':
        raise ValueError('The sklearn baseline supports Euclidean only; select scratch for Mahalanobis.')
    distances = ['euclidean', 'mahalanobis'] if impl == 'scratch' else ['euclidean']
    runs = {}
    for distance in distances:
        run_cfg = copy.deepcopy(cfg)
        run_cfg['kmeans']['distance'] = distance
        run_cfg['output']['dir'] = str(out / distance)
        runs[distance] = run_experiment(run_cfg, data)
    # Keep top-level Euclidean fields for older consumers, with explicit run records.
    summary = dict(runs['euclidean'])
    summary.update(schema_version=2, mode='comparison' if impl == 'scratch' else 'baseline',
                   runs=runs, comparison_note='Same sample, k, seed and restart budget. Do not compare inertia across distances.')
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    if impl != 'scratch':
        print('Baseline only: implement the student stubs and select scratch to reproduce both distances.')
    elif cfg['kmeans'].get('mahalanobis_diag') is None:
        print('Identity weights: select and justify a positive diagonal for the Mahalanobis comparison.')
    print('Combined metrics:', summary_path)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    main(args.config, args.output_dir)
