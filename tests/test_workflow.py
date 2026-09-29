"""Provided orchestration and reference checks; student algorithm stubs stay untouched."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from src import reproduce, cluster, evaluate, download_data
from src.kmeans import build
from grading import grade


def config(tmp_path, impl='sklearn'):
    return {'seed':42, 'data':{'source':'iris'},
            'kmeans':{'implementation':impl,'k':2,'n_init':10,'init':'kmeans++',
                      'max_iter':300,'tol':1e-4,'distance':'euclidean'},
            'selection':{'k_min':2,'k_max':2}, 'compare_to_reference':True,
            'output':{'dir':str(tmp_path / 'out')}}


def test_sklearn_rejects_weighted_request(tmp_path):
    cfg=config(tmp_path);cfg['kmeans']['distance']='mahalanobis'
    with pytest.raises(ValueError, match='Euclidean only'):
        build(cfg)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='Euclidean only'):
        reproduce.main(path)


def test_weighted_reference_same_objective(tmp_path, monkeypatch):
    X=np.array([[0,0],[0,1],[1,0],[10,10],[10,11],[11,10]], dtype=float)
    labels=np.array([0,0,0,1,1,1]);centroids=np.array([X[:3].mean(0),X[3:].mean(0)])
    # Fixed known partition isolates supplied reference/evaluation logic from student work.
    inertia=float(((X-centroids[labels])**2 * 100).sum())
    est=SimpleNamespace(fit=lambda X:est,labels_=labels,centroids_=centroids,inertia_=inertia,n_iter_=1)
    monkeypatch.setattr(cluster.km,'build',lambda cfg:est)
    cfg=config(tmp_path,'scratch');cfg['kmeans'].update(distance='mahalanobis',mahalanobis_diag=[100,100])
    m=cluster.run_experiment(cfg,(X,labels,['a','b'],['left','right']))
    assert m['reference_check']['inertia_ratio']==pytest.approx(1)
    assert m['reference_check']['geometry']=='mahalanobis'
    assert m['silhouette_euclidean']==pytest.approx(m['silhouette_configured'])
    assert m['configuration']['kmeans']['mahalanobis_diag']==[100,100]


def test_silhouette_labels_and_degenerate_partition():
    X=np.array([[0,0],[0,2],[1,0],[8,0],[8,2],[9,0]],float)
    m=evaluate.internal_metrics(X,np.array([0,0,0,1,1,1]),1,X*[.1,10])
    assert m['silhouette']==m['silhouette_euclidean']
    assert m['silhouette_euclidean']!=pytest.approx(m['silhouette_configured'])
    assert evaluate.internal_metrics(X,np.zeros(6),0)['silhouette'] is None


def test_comparison_uses_same_data_and_preserves_config(tmp_path,monkeypatch):
    cfg=config(tmp_path,'scratch');cfg['kmeans']['mahalanobis_diag']=[1,2,3,4]
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg));calls=[]
    def run(c,data):
        calls.append((copy.deepcopy(c),data))
        return {'distance':c['kmeans']['distance']}
    monkeypatch.setattr(reproduce,'run_experiment',run)
    m=reproduce.main(path)
    assert set(m['runs'])=={'euclidean','mahalanobis'}
    assert calls[0][1] is calls[1][1]
    assert calls[0][0]['seed']==calls[1][0]['seed']
    assert calls[0][0]['kmeans']['k']==calls[1][0]['kmeans']['k']
    assert yaml.safe_load(path.read_text())==cfg
    assert (tmp_path/'out/metrics.json').exists()


def test_failed_comparison_removes_old_summary(tmp_path,monkeypatch):
    cfg=config(tmp_path,'scratch');path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    out=tmp_path/'out';out.mkdir();(out/'metrics.json').write_text('{}')
    def fail(*args):raise NotImplementedError('student stub')
    monkeypatch.setattr(reproduce,'run_experiment',fail)
    with pytest.raises(NotImplementedError):reproduce.main(path)
    assert not (out/'metrics.json').exists()


def test_grading_executes_make_in_fresh_directory(tmp_path,monkeypatch):
    cfg=config(tmp_path);path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    (tmp_path/'out').mkdir();(tmp_path/'out/metrics.json').write_text('{}')
    monkeypatch.setattr(grade,'ROOT',tmp_path)
    monkeypatch.setattr(grade,'_REPRODUCE_TEMP',None)
    monkeypatch.setattr(grade,'_REPRODUCE_OK',None)
    calls=[]
    def run(cmd,**kw):calls.append(cmd);return 0,'ok'
    monkeypatch.setattr(grade,'run',run)
    try:
        assert grade.check_reproduce()['status']=='pass'
        assert calls[0][:2]==['make','reproduce']
        output=Path(next(x.split('=',1)[1] for x in calls[0] if x.startswith('OUTPUT_DIR=')))
        assert output!=tmp_path/'out'
        assert grade.check_metrics()['status']=='fail'
    finally:
        grade._REPRODUCE_TEMP.cleanup()


def test_downloader_verifies_and_does_not_overwrite(tmp_path,monkeypatch):
    content=b'example training CSV'
    import hashlib
    monkeypatch.setattr(download_data,'SHA256',hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(download_data,'urlopen',lambda *a,**kw:io.BytesIO(content))
    path=tmp_path/'train.csv'
    download_data.download(path)
    assert path.read_bytes()==content
    path.write_bytes(b'user data')
    with pytest.raises(ValueError,match='not overwritten'):download_data.download(path)
    assert path.read_bytes()==b'user data'


def test_bad_download_leaves_no_destination(tmp_path,monkeypatch):
    monkeypatch.setattr(download_data,'urlopen',lambda *a,**kw:io.BytesIO(b'wrong'))
    with pytest.raises(ValueError,match='checksum'):download_data.download(tmp_path/'train.csv')
    assert not list(tmp_path.iterdir())


def test_comparison_requires_both_runs_and_real_outputs(tmp_path, monkeypatch):
    cfg=config(tmp_path,'scratch')
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr(grade,'ROOT',tmp_path)
    monkeypatch.setattr(grade,'_REPRODUCE_TEMP',None)
    monkeypatch.setattr(grade,'_REPRODUCE_OK',None)
    summary={'runs':{'euclidean':{'distance':'euclidean','implementation':'scratch'}}}
    (out/'metrics.json').write_text(json.dumps(summary))
    assert grade.check_comparison()['status']=='fail'
    summary['runs']['mahalanobis']={'distance':'mahalanobis','implementation':'scratch','configuration':None}
    (out/'metrics.json').write_text(json.dumps(summary))
    assert grade.check_comparison()['status']=='fail'


def test_nested_invalid_metric_cannot_hide_behind_valid_summary(tmp_path,monkeypatch):
    cfg=config(tmp_path);(tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr(grade,'ROOT',tmp_path)
    monkeypatch.setattr(grade,'_REPRODUCE_TEMP',None)
    monkeypatch.setattr(grade,'_REPRODUCE_OK',None)
    good=dict(k=2,inertia=1,silhouette=.5,v_measure=.5,accuracy=.5,f1=.5)
    record=dict(good,f1=None)
    (out/'metrics.json').write_text(json.dumps(dict(good,runs={'mahalanobis':record})))
    assert grade.check_metrics()['status']=='fail'
