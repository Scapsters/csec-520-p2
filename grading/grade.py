#!/usr/bin/env python3
"""Automated grading harness for CSEC 520/620 — Project 2 (K-means).

Runs the OBJECTIVE checks a grading agent needs as evidence, then writes
grading/auto_report.json. The agent reads that file, adds qualitative judgment
(discussion quality, correctness, code quality), and produces the final grade
per grading/AGENT_GRADING.md.

Requires PyYAML (included in requirements.txt). Run from the repo root:  python grading/grade.py
"""
from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["config.yaml", "src/cluster.py", "src/data.py", "src/kmeans.py",
            "src/distances.py", "src/evaluate.py", "README.md", "requirements.txt"]
# Clustering metrics we expect in results/metrics.json.
METRIC_KEYS = {"k", "inertia", "silhouette", "v_measure", "accuracy", "f1"}
UNIT_INTERVAL = {"silhouette", "homogeneity", "completeness", "v_measure",
                 "nmi", "accuracy", "precision", "recall", "f1",
                 "precision_weighted", "recall_weighted", "f1_weighted",
                 "silhouette_euclidean", "silhouette_configured"}

_venv = ROOT / ".venv" / "bin" / "python"
PYEXE = str(_venv) if _venv.exists() else sys.executable
_REPRODUCE_TEMP = None
_REPRODUCE_OK = None


def run(cmd, timeout=900):
    """Run a command from ROOT; return (returncode, tail_of_output)."""
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return 127, str(e)


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def check_structure():
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    return {"id": "structure", "status": "pass" if not missing else "fail",
            "evidence": "all present" if not missing else f"missing: {missing}"}


def check_tests():
    rc, out = run([PYEXE, "-m", "pytest", "-q"], timeout=300)
    status = "pass" if rc == 0 else ("dep_missing" if "No module named" in out else "fail")
    return {"id": "tests", "status": status, "returncode": rc, "evidence": out[-800:]}


def check_reproduce():
    global _REPRODUCE_TEMP, _REPRODUCE_OK
    if _REPRODUCE_TEMP is not None:
        _REPRODUCE_TEMP.cleanup()
    _REPRODUCE_TEMP = tempfile.TemporaryDirectory(prefix="p2-grading-")
    rc, out = run(["make", "reproduce", f"PY={PYEXE}",
                   f"OUTPUT_DIR={_REPRODUCE_TEMP.name}"])
    _REPRODUCE_OK = rc == 0
    if rc == 0:
        status = "pass"
    elif "NotImplementedError" in out:
        status = "fail"          # stubs never filled in
    elif "No module named" in out:
        status = "dep_missing"
    else:
        status = "fail"
    return {"id": "reproduce_runs", "status": status, "returncode": rc,
            "evidence": out[-1200:]}


def _metrics():
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        output_dir = cfg["output"]["dir"]
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output.dir must be a non-empty path string")
        if _REPRODUCE_OK is False:
            return None, "reproduction failed; old metrics are not accepted"
        p = (Path(_REPRODUCE_TEMP.name) if _REPRODUCE_TEMP is not None
             else ROOT / output_dir) / "metrics.json"
        m = json.loads(p.read_text())
        if not isinstance(m, dict):
            raise ValueError("metrics.json must contain a JSON object")
        return m, None
    except Exception as e:
        return None, f"cannot read configured metrics: {e}"


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _validate_metrics(m):
    if not isinstance(m, dict):
        return {"error": "metrics must be an object"}
    missing = METRIC_KEYS - set(m)
    bad = {}
    for k in (METRIC_KEYS | UNIT_INTERVAL | {"adjusted_rand"}) & m.keys():
        v = m[k]
        valid = _number(v)
        if valid:
            if k == "k":
                valid = type(v) is int and v >= 1
            elif k == "inertia":
                valid = v >= 0
            elif k in {"silhouette", "silhouette_euclidean", "silhouette_configured", "adjusted_rand"}:
                valid = -1 <= v <= 1
            else:
                valid = 0 <= v <= 1
        if not valid:
            bad[k] = repr(v)
    ok = not missing and not bad
    head = {k: m[k] for k in sorted(METRIC_KEYS & set(m))}
    return {} if ok else {"metrics": head, "missing_keys": sorted(missing), "invalid_values": bad}


def check_metrics():
    m, err = _metrics()
    if m is None:
        return {"id": "metrics_present", "status": "fail", "evidence": err}
    errors = {"summary": _validate_metrics(m)}
    runs = m.get("runs", {})
    if not isinstance(runs, dict):
        errors["runs"] = "runs must be an object"
    else:
        errors.update({name: _validate_metrics(record) for name, record in runs.items()})
    errors = {name: error for name, error in errors.items() if error}
    return {"id": "metrics_present", "status": "fail" if errors else "pass",
            "evidence": errors or "numeric metrics valid in summary and every recorded run"}


def check_comparison():
    m, err = _metrics()
    if m is None:
        return {"id": "comparison_complete", "status": "fail", "evidence": err}
    runs = m.get("runs")
    errors = []
    if not isinstance(runs, dict) or set(runs) != {"euclidean", "mahalanobis"}:
        errors.append("both euclidean and mahalanobis runs are required for submission")
    else:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        out = Path(_REPRODUCE_TEMP.name) if _REPRODUCE_TEMP is not None else ROOT / cfg['output']['dir']
        for name, record in runs.items():
            if not isinstance(record, dict) or record.get('distance') != name or record.get('implementation') != 'scratch':
                errors.append(f"{name}: expected a correctly labeled scratch run")
                continue
            for key in ['silhouette_euclidean', 'silhouette_configured']:
                v = record.get(key)
                if not _number(v) or not -1 <= v <= 1:
                    errors.append(f"{name}: invalid {key}")
            for filename in ['k_selection.png', 'clusters.png', 'confusion_matrix.png', 'metrics.json']:
                path = out / name / filename
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"{name}: missing {filename}")
                elif filename.endswith('.png'):
                    with path.open('rb') as image:
                        if image.read(8) != b'\x89PNG\r\n\x1a\n':
                            errors.append(f"{name}: invalid PNG {filename}")
                else:
                    try:
                        if json.loads(path.read_text()) != record:
                            errors.append(f"{name}: per-run metrics differ from combined metrics")
                    except (ValueError, OSError):
                        errors.append(f"{name}: invalid per-run JSON")
            config = record.get('configuration', {})
            if not isinstance(config, dict) or not isinstance(config.get('kmeans'), dict):
                errors.append(f"{name}: missing configuration")
                continue
            expected = cfg.get('kmeans', {})
            recorded = config['kmeans']
            if config.get('seed') != cfg['seed'] or config.get('data') != cfg['data']:
                errors.append(f"{name}: data/seed differs from config.yaml")
            weights = record.get('diagonal_weights')
            features = record.get('feature_names')
            if (not isinstance(weights, list) or not isinstance(features, list) or
                    len(weights) != len(features) or not features or
                    not all(_number(v) and v > 0 for v in weights)):
                errors.append(f"{name}: invalid weights or feature order")
            elif name == 'euclidean' and any(v != 1 for v in weights):
                errors.append(f"{name}: expected identity weights")
            elif name == 'mahalanobis':
                specified = expected.get('mahalanobis_diag')
                if specified is not None and weights != specified:
                    errors.append(f"{name}: weights differ from config.yaml")
            for key in ['k', 'init', 'n_init', 'max_iter', 'tol']:
                if recorded.get(key) != expected.get(key):
                    errors.append(f"{name}: {key} differs from config.yaml")
    return {"id": "comparison_complete", "status": "fail" if errors else "pass",
            "evidence": errors or "both scratch runs and figures reproduced with matched settings"}


def _scratch_class():
    """Return the ast.ClassDef for KMeansScratch, or None."""
    try:
        tree = ast.parse(_read(ROOT / "src" / "kmeans.py"))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KMeansScratch":
            return node
    return None


def check_scratch_implemented():
    """The four core pieces must no longer be NotImplementedError stubs."""
    cls = _scratch_class()
    if cls is None:
        return {"id": "scratch_implemented", "status": "fail",
                "evidence": "KMeansScratch class not found in src/kmeans.py"}
    wanted = {"_init_centroids", "_assign", "_update", "fit"}
    stubs, found = [], set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            found.add(node.name)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Raise):
                    exc = sub.exc
                    name = getattr(exc, "id", None) or getattr(
                        getattr(exc, "func", None), "id", None)
                    if name == "NotImplementedError":
                        stubs.append(node.name)
    missing = sorted(wanted - found)
    ok = not stubs and not missing
    return {"id": "scratch_implemented", "status": "pass" if ok else "fail",
            "evidence": {"still_stubs": sorted(set(stubs)), "methods_missing": missing}}


def check_no_library_kmeans_in_scratch():
    """KMeansScratch must not delegate to a library K-means."""
    cls = _scratch_class()
    if cls is None:
        return {"id": "no_library_kmeans_in_scratch", "status": "fail",
                "evidence": "KMeansScratch class not found"}
    hits = []
    for node in ast.walk(cls):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = ",".join(a.name for a in node.names)
            if "sklearn" in mod or "scipy.cluster" in mod or "KMeans" in names:
                hits.append(f"import {mod}.{names}")
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name in {"KMeans", "MiniBatchKMeans", "kmeans", "kmeans2", "vq"}:
                hits.append(f"call {name}()")
    return {"id": "no_library_kmeans_in_scratch",
            "status": "clean" if not hits else "violation",
            "evidence": hits or "no library clustering call inside KMeansScratch"}


def check_config_uses_scratch():
    try:
        import yaml
        impl = yaml.safe_load((ROOT / "config.yaml").read_text())["kmeans"]["implementation"]
    except Exception as e:
        return {"id": "config_uses_scratch", "status": "fail", "evidence": str(e)}
    return {"id": "config_uses_scratch", "status": "pass" if impl == "scratch" else "fail",
            "evidence": f"kmeans.implementation = {impl!r} (must be scratch for submission)"}


def check_reference_agreement():
    """If the run recorded a reference check, report how close scratch came."""
    m, err = _metrics()
    if m is None:
        return {"id": "reference_agreement", "status": "review", "evidence": err}
    runs = m.get("runs", {"single": m})
    if not isinstance(runs, dict):
        return {"id": "reference_agreement", "status": "review", "evidence": "invalid runs"}
    evidence = {}
    for name, record in runs.items():
        rc = record.get("reference_check") if isinstance(record, dict) else None
        if not isinstance(rc, dict):
            evidence[name] = "reference check missing; enable compare_to_reference"
            continue
        ratio, ari = rc.get("inertia_ratio"), rc.get("ari_vs_reference")
        comparable = rc.get('geometry') == record.get('distance')
        ok = comparable and ((_number(ratio) and 0 <= ratio <= 1.10) or
                             (_number(ari) and 0.80 <= ari <= 1))
        evidence[name] = {"status": "pass" if ok else "review", "reference_check": rc}
    ok = bool(evidence) and all(isinstance(v, dict) and v['status'] == 'pass' for v in evidence.values())
    return {"id": "reference_agreement", "status": "pass" if ok else "review", "evidence": evidence}


def check_mahalanobis_implemented():
    src = _read(ROOT / "src" / "distances.py")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"id": "mahalanobis_implemented", "status": "fail",
                "evidence": "src/distances.py does not parse"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "mahalanobis_sqdist":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Raise):
                    exc = sub.exc
                    name = getattr(exc, "id", None) or getattr(
                        getattr(exc, "func", None), "id", None)
                    if name == "NotImplementedError":
                        return {"id": "mahalanobis_implemented", "status": "fail",
                                "evidence": "mahalanobis_sqdist is still a stub"}
            return {"id": "mahalanobis_implemented", "status": "pass",
                    "evidence": "mahalanobis_sqdist has a real body"}
    return {"id": "mahalanobis_implemented", "status": "fail",
            "evidence": "mahalanobis_sqdist not found"}


def check_label_leakage():
    """Clustering must not receive labels. Heuristic — agent should confirm."""
    hits = []
    patterns = [r"\.fit\s*\([^)]*\by\b[^)]*\)", r"fit\s*\(\s*X\s*,\s*y", r"labels\s*=\s*y\b"]
    for pyf in sorted((ROOT / "src").glob("*.py")):
        text = _read(pyf)
        for pat in patterns:
            for m in re.finditer(pat, text):
                hits.append(f"{pyf.name}: {m.group(0).strip()}")
    return {"id": "label_leakage_scan", "status": "review" if hits else "clean",
            "evidence": hits or "no sign of labels passed into clustering (heuristic only)"}


def check_report():
    rdir = ROOT / "report"
    files = ([p.name for p in rdir.glob("**/*")
              if p.suffix.lower() in {".tex", ".pdf", ".md"}] if rdir.exists() else [])
    has_real = any(f.lower().endswith((".tex", ".pdf")) for f in files)
    return {"id": "report_present", "status": "pass" if has_real else "review",
            "evidence": f"report files: {files}" if files else "no report/ .tex or .pdf found"}


def check_git_hygiene():
    txt = _read(ROOT / ".gitignore")
    committed = ([p.name for p in (ROOT / "data").glob("*")
                  if p.name not in {"README.md", ".gitkeep", "iris"}]
                 if (ROOT / "data").exists() else [])
    ok = "data/" in txt and "results/" in txt
    return {"id": "git_hygiene", "status": "pass" if ok else "review",
            "evidence": {"gitignore_covers_data_results": ok,
                         "possible_committed_data": committed}}


def main():
    checks = [check_structure(), check_tests(), check_reproduce(), check_metrics(), check_comparison(),
              check_config_uses_scratch(), check_scratch_implemented(),
              check_no_library_kmeans_in_scratch(), check_reference_agreement(),
              check_mahalanobis_implemented(), check_label_leakage(),
              check_report(), check_git_hygiene()]
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo": ROOT.name,
        "project": "Project 2 — K-means Clustering",
        "checks": checks,
        "note_for_agent": (
            "These are OBJECTIVE signals only. status 'dep_missing' means the grader "
            "environment lacked a dependency — set up the env and re-run before "
            "penalizing reproducibility. 'scratch_implemented' and "
            "'no_library_kmeans_in_scratch' are the key integrity checks for this "
            "project: the student must write Lloyd's algorithm themselves. Confirm "
            "heuristic flags by reading the code, then score every rubric criterion "
            "per AGENT_GRADING.md."
        ),
    }
    out = ROOT / "grading" / "auto_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
