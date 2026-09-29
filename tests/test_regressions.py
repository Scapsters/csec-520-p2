"""Regression coverage for the provided pipeline, not student algorithms."""
import json
import numpy as np
import pandas as pd
import pytest
import yaml
from grading import grade
from src.data import load_data
from src.cluster import main
from src.evaluate import save_confusion_matrix


@pytest.fixture
def metrics_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, 'ROOT', tmp_path)
    (tmp_path / 'config.yaml').write_text('output: {dir: custom}\n')
    (tmp_path / 'custom').mkdir()
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/metrics.json').write_text('{}')
    return tmp_path


def valid_metrics():
    return dict(k=3, inertia=12.0, silhouette=-.2, v_measure=.8, accuracy=.8, f1=.8)


def test_custom_directory_and_negative_silhouette(metrics_repo):
    (metrics_repo / 'custom/metrics.json').write_text(json.dumps(valid_metrics()))
    assert grade.check_metrics()['status'] == 'pass'
    (metrics_repo / 'custom/metrics.json').unlink()
    (metrics_repo / 'results/metrics.json').write_text(json.dumps(valid_metrics()))
    assert grade.check_metrics()['status'] == 'fail'


@pytest.mark.parametrize('key,value', [
    ('f1', None), ('f1', '0.9'), ('f1', True), ('accuracy', -.1), ('f1', 1.1),
    ('silhouette', float('nan')), ('inertia', float('inf')), ('inertia', -1),
    ('k', 0), ('k', 2.5), ('k', True), ('v_measure', []), ('f1', {}),
])
def test_invalid_metrics(metrics_repo, key, value):
    m = valid_metrics(); m[key] = value
    (metrics_repo / 'custom/metrics.json').write_text(json.dumps(m))
    assert grade.check_metrics()['status'] == 'fail'


@pytest.mark.parametrize('value', [[], None, 1, 'bad', {}])
def test_bad_metrics_shape_or_keys(metrics_repo, value):
    (metrics_repo / 'custom/metrics.json').write_text(json.dumps(value))
    assert grade.check_metrics()['status'] == 'fail'
    grade.check_reference_agreement()  # must not crash the rest of the harness


def test_bad_reference_check(metrics_repo):
    m = valid_metrics(); m['reference_check'] = {'inertia_ratio': 'bad', 'ari_vs_reference': True}
    (metrics_repo / 'custom/metrics.json').write_text(json.dumps(m))
    assert grade.check_reference_agreement()['status'] == 'review'


def test_csv_multiclass_single_feature_pipeline(tmp_path):
    frame = pd.DataFrame({'value': np.arange(30), 'label': np.repeat(['attack', 'benign', 'other'], 10)})
    path = tmp_path / 'data.csv'; frame.to_csv(path, index=False)
    cfg = {'seed': 42, 'data': {'source': 'csv', 'csv_path': str(path), 'target': 'label'},
           'kmeans': {'implementation': 'sklearn', 'k': 3, 'n_init': 2},
           'selection': {'k_min': 2, 'k_max': 4}, 'output': {'dir': str(tmp_path / 'out')}}
    X, y, features, names = load_data(cfg)
    assert names == ['attack', 'benign', 'other']
    assert np.array_equal(y, np.repeat([0, 1, 2], 10))
    config = tmp_path / 'config.yaml'; config.write_text(yaml.safe_dump(cfg))
    main(str(config))
    for filename in ['metrics.json', 'clusters.png', 'confusion_matrix.png', 'k_selection.png']:
        assert (tmp_path / 'out' / filename).stat().st_size > 0
    frame.loc[0, 'label'] = None; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='missing labels'):
        load_data(cfg)


def test_confusion_matrix_preserves_absent_class(tmp_path):
    cm = save_confusion_matrix(np.array([0, 2]), np.array([0, 2]), tmp_path / 'cm.png', ['a', 'b', 'c'])
    assert cm == [[1, 0, 0], [0, 0, 0], [0, 0, 1]]


@pytest.mark.parametrize('values,message', [([float('inf')] * 6, 'finite rows'), ([1] * 6, 'non-constant')])
def test_unusable_csv_features(tmp_path, values, message):
    path = tmp_path / 'data.csv'
    pd.DataFrame({'x': values, 'target': ['a', 'b'] * 3}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_data({'seed': 42, 'data': {'source': 'csv', 'csv_path': str(path), 'target': 'target'}})
