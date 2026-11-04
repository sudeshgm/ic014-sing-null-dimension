"""Tests of the standalone check (sanity_check_general.py) with user score functions; synthetic data only."""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R9 = os.path.dirname(HERE)
sys.path.insert(0, R9)
import sanity_check_general as G


def test_energy_score_is_flagged():
    out = G.run_check(G.toy_models(n=12, seed=1), G.null_share_score, heads=False)
    assert out['verdict'] == 'ENERGY-DRIVEN'


def test_planted_content_is_not_flagged():
    n = 12; c = np.random.default_rng(2).permutation(np.linspace(0, 0.9, n))
    models = G.toy_models(n=n, energy=np.ones(n), content=c, seed=2)

    def content_score(m):  # mean cosine of null coordinates with the fit-half class mean of the true class
        Vn = G.default_subspace(m)[0]; Z = np.asarray(m['F'], float) @ Vn; fit = m['fit']
        M = G.class_means(Z[fit], m['y'][fit], int(m['y'].max()) + 1)
        e = ~fit; a = Z[e] - Z[fit].mean(0); p = M[m['y'][e]] - Z[fit].mean(0)
        return float(np.mean(np.sum(a * p, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(p, axis=1)))
    out = G.run_check(models, content_score, heads=False)
    assert out['verdict'] != 'ENERGY-DRIVEN'
    assert out['step2']['tau_E']['tau'] < 0.5


def test_custom_subspace_and_dict_score():
    models = G.toy_models(n=8, seed=3)

    def pca_residual(m):  # residual of the top-8 principal directions of the fit images
        F = np.asarray(m['F'], float)[m['fit']]; mu = F.mean(0)
        V = np.linalg.svd(F - mu, full_matrices=False)[2].T
        return V[:, 8:], mu

    def score(m):
        V, mu = pca_residual(m); R = (np.asarray(m['F'], float) - mu) @ V
        return {'res_norm': float(np.mean(np.linalg.norm(R, axis=1))), 'res_var': float(np.mean(R ** 2))}
    out = G.run_check(models, score, subspace_fn=pca_residual, heads=False)
    assert set(out['scores']) == {'res_norm', 'res_var'}
    assert all(o['verdict'] in ('ENERGY-DRIVEN', 'MIXED', 'CONTENT-READING') for o in out['scores'].values())


def test_step3_heads_reported_with_top1():
    out = G.run_check(G.toy_models(n=8, seed=4), G.null_energy_score, heads=True)
    s3 = out['step3']
    assert set(s3) == {'NCM', 'LDA', 'ridge', 'perturbed'}
    for d in s3.values():
        assert d['n'] == 8 and 0 <= d['n_matched'] <= 8 and len(d['top1_gap_pts']) == 8
    t = out['per_model']['toy0']['heads']
    assert all(0 <= h['top1'] <= 1 for h in t.values()) and t['released']['matched']


def test_cli_with_user_score_file(tmp_path):
    for k, m in G.toy_models(n=8, seed=5).items():
        np.savez(tmp_path / f'{k}.npz', F=m['F'].astype(np.float32), y=m['y'], W=m['W'], b=m['b'])
    (tmp_path / 'my_score.py').write_text(
        'import numpy as np\n'
        'def score(m):\n'
        '    W = m["W"]; P = np.linalg.pinv(W) @ W\n'
        '    F = np.asarray(m["F"], float); N = F - F @ P\n'
        '    return float(np.mean(np.linalg.norm(N, axis=1)))\n')
    cmd = [sys.executable, os.path.join(R9, 'sanity_check_general.py'), '--npz', *sorted(str(p) for p in tmp_path.glob('toy*.npz')),
           '--score', f'{tmp_path / "my_score.py"}:score', '--out', str(tmp_path / 'o.json')]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    o = json.load(open(tmp_path / 'o.json'))
    assert o['verdict'] in ('ENERGY-DRIVEN', 'MIXED', 'CONTENT-READING') and 'step3' in o


def test_copy_alone_runs_outside_repo(tmp_path):
    import shutil
    shutil.copy(os.path.join(R9, 'sanity_check_general.py'), tmp_path)
    code = ('import sys, sanity_check_general as G; G.main(["--demo", "--no-heads"]); '
            'bad = [m for m in sys.modules if m.split(".")[0] in ("scores_r9", "controls_r9", "run_r9", "torch", "pandas")]; '
            'assert not bad, bad')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
