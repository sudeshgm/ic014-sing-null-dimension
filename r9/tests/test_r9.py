"""Round 9 unit tests (synthetic data only)."""
import ast, json, os, sys
import numpy as np
import pytest
from scipy.stats import kendalltau

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import sanity_check_general as G
import scores_r9 as S
import controls_r9 as C
import run_r9 as R


@pytest.fixture(scope='module')
def toy():
    return R.synthetic_models(n=6, d=80, K=10, per_class=10, content=np.linspace(0, 0.6, 6))


def test_general_tool_imports_only_numpy_scipy_sklearn():
    tree = ast.parse(open(os.path.join(HERE, 'sanity_check_general.py')).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): mods |= {a.name.split('.')[0] for a in n.names}
        if isinstance(n, ast.ImportFrom): mods.add(n.module.split('.')[0])
    stdlib = {'copy', 'importlib', 'os', 'argparse', 'json'}  # standard library, command line only
    assert mods <= {'numpy', 'scipy', 'sklearn'} | stdlib, mods


def test_ktau_rows_matches_scipy():
    r = np.random.default_rng(0); X = r.standard_normal((50, 9)); Y = X + r.standard_normal((50, 9)); Y[3, :4] = Y[3, 0]
    t = G.ktau_rows(X, Y)
    assert np.allclose(t, [kendalltau(x, y).statistic for x, y in zip(X, Y)])


def test_null_basis_and_offset():
    r = np.random.default_rng(1); W = r.standard_normal((10, 30)); b = r.standard_normal(10)
    V = G.null_basis(W); assert V.shape == (30, 20) and np.abs(W @ V).max() < 1e-10
    o = S.head_offset(W, b); assert np.allclose(W @ o, b) and np.abs(V.T @ o).max() < 1e-10


def test_rho_decomposition_identity(toy):
    m = toy['m2']; V = G.null_basis(m['W']); d = G.rho_decomposition(m['F'], m['y'], V)
    assert d['identity_abs_err'] < 1e-8 and abs(d['rho'] - (d['rho_offset'] + d['rho_between'] + d['rho_within'])) < 1e-8


def test_fast_paths_equal_generic(toy):
    raw = toy['m3']; mn, mv, info = R.prepare(raw); g = R.generic(raw); g['Vn'] = G.null_basis(raw['W'])
    a, b = S.null_family(mn), S.null_family(g)
    for k in a: assert abs(a[k] - b[k]) < 1e-4 * max(1, abs(b[k])), (k, a[k], b[k])
    a, b = S.vim_family(mv), S.vim_family(g)
    for k in a: assert abs(a[k] - b[k]) < 1e-3 * max(1, abs(b[k])), (k, a[k], b[k])
    # direct-coordinate path used by the controls equals replacing the features
    r = np.random.default_rng(2); s = mn['_null_from']; Cn = r.standard_normal((len(raw['y']), info['null_dim'])).astype(np.float32)
    m1 = dict(mn); m1['_N'] = Cn; m2 = dict(mn); m2['F'] = G.replace(mn['F'], slice(s, None), Cn)
    a, b = S.null_family(m1), S.null_family(m2)
    for k in a: assert abs(a[k] - b[k]) < 1e-4 * max(1, abs(b[k])), k
    D = mv['_vim_frozen']; Cr = r.standard_normal((len(raw['y']), info['vim_res_dim'])).astype(np.float32)
    m1 = dict(mv); m1['_R'] = Cr; m2 = dict(mv); m2['F'] = G.replace(mv['F'], slice(D, None), Cr)
    m2 = {k: v for k, v in m2.items() if k not in ('_pn2', '_Lp', '_WR')}
    a, b = S.vim_family(m1), S.vim_family(m2)
    for k in a: assert abs(a[k] - b[k]) < 1e-3 * max(1, abs(b[k])), (k, a[k], b[k])


def test_nsp_definition(toy):
    m = dict(toy['m1']); m['fit'] = G.split_first_half(m['y']); m['F'] = m['F'].astype(np.float64)
    V = G.null_basis(m['W']); o = np.linalg.pinv(m['W']) @ m['b']; ev = ~m['fit']; F = m['F'][ev]
    ref = np.mean(np.linalg.norm(F @ V, axis=1) / np.linalg.norm(F + o, axis=1))
    assert abs(S.null_family(m)['nsp'] - ref) < 1e-10


def test_controls_moments():
    r = np.random.default_rng(3); k, N, K = 20, 4000, 10; y = np.repeat(np.arange(K), N // K)
    Cr = r.standard_normal((N, k)) @ r.standard_normal((k, k)) + 2.0; mu, L = G.moment_sqrt(Cr); Ec = np.sum(L ** 2)
    nc = C.make_control(Cr, y, 'NC', 2.0, 0.5, 0.0, None, np.random.default_rng(4), mu, L)
    assert np.allclose(nc.mean(0), 0.5 * mu, atol=0.15 * np.sqrt(Ec / k))
    assert abs(np.sum((nc - nc.mean(0)) ** 2) / N / (2.0 * Ec) - 1) < 0.05
    U = C.random_prototypes(K, k, np.random.default_rng(5))
    pc = C.make_control(Cr, y, 'PCm', 1.0, 1.0, 0.6, U, np.random.default_rng(6), mu, L)
    assert abs(np.sum((pc - mu) ** 2) / N / Ec - 1) < 0.1                 # energy kept
    M = G.class_means(pc - mu, y, K); cos = np.sum(M * U, 1) / np.linalg.norm(M, axis=1)
    assert cos.mean() > 0.9                                                 # planted content recoverable


def test_dataset_params_deterministic():
    a = C.dataset_params(2, 'PCm', 4, 7, 13); b = C.dataset_params(2, 'PCm', 4, 7, 13)
    assert np.array_equal(a['a'], b['a']) and np.isclose(a['a'].max(), C.A_LEVELS[4]) and np.all(a['e'] == 1)
    n = C.dataset_params(1, 'NC', 0, 0, 13); assert set(n['e']) <= set(C.E_GRID) and set(n['o']) <= set(C.O_GRID)


def test_check_flags_energy_and_not_content():
    # energy-only score on energy-only models: ENERGY-DRIVEN
    ms = R.synthetic_models(n=10, d=60, K=8, per_class=10, energy=np.linspace(0.3, 3, 10), seed=1, translator=False)
    for m in ms.values(): m['fit'] = G.split_first_half(m['y'])
    out = G.run_check(ms, G.null_energy_score, draws=2, heads=False, B=500)
    assert out['verdict'] == 'ENERGY-DRIVEN'
    # content score on models whose null content varies at fixed energy: not ENERGY-DRIVEN
    ms = R.synthetic_models(n=10, d=60, K=8, per_class=10, energy=np.ones(10), content=np.random.default_rng(0).permutation(np.linspace(0, 0.8, 10)),
                            seed=2, translator=False)
    for m in ms.values(): m['fit'] = G.split_first_half(m['y'])
    out = G.run_check(ms, lambda m: S.null_family(m)['core_null'], draws=2, heads=False, B=500)
    assert out['verdict'] != 'ENERGY-DRIVEN'


def test_heads_and_dict_scores():
    ms = R.synthetic_models(n=8, d=60, K=8, per_class=10, seed=3, translator=False)
    for m in ms.values(): m['fit'] = G.split_first_half(m['y'])
    out = G.run_check(ms, lambda m: {k: v for k, v in S.null_family(m).items() if k in ('nsp', 'rho')}, draws=2, B=300)
    assert set(out['scores']) == {'nsp', 'rho'}
    st3 = out['scores']['rho']['step3']; assert {'NCM', 'LDA', 'ridge', 'perturbed'} <= set(st3)
    H = G.fit_heads(ms['m0']); assert H['ridge']['matched'] and abs(H['perturbed']['top1'] - H['released']['top1']) <= 0.02


def test_with_head_drops_caches(toy):
    mn, mv, _ = R.prepare(toy['m0']); h = G.with_head(mn, np.eye(10, 80), np.zeros(10))
    assert not any(k.startswith('_') for k in h) and 'Vn' not in h


def test_runner_end_to_end_synthetic(tmp_path):
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, 'run_r9.py'), '--synthetic', '--syn-n', '6', '--syn-d', '48', '--syn-K', '6',
                        '--out', str(tmp_path), '--reps-scale', '0.05'], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    for f in ('r9_controls.json', 'r9_real_checks.json', 'r9_head_swap.json', 'r9_per_model.json'):
        assert (tmp_path / f).exists()


def test_parallel_workers_match_single_run(tmp_path):
    """--only workers sharing a work folder give the same per-model results as one process over the full list."""
    import subprocess
    run = os.path.join(HERE, 'run_r9.py'); base = [sys.executable, run, '--synthetic', '--syn-n', '6', '--syn-d', '48',
                                                    '--syn-K', '6', '--reps-scale', '0.05', '--no-heads']
    one, par = tmp_path / 'one', tmp_path / 'par'
    subprocess.run(base + ['--out', str(one)], check=True, capture_output=True)
    for names in (['m0', 'm2', 'm4'], ['m1', 'm3', 'm5']):
        subprocess.run(base + ['--out', str(par), '--only', *names], check=True, capture_output=True)
    subprocess.run(base + ['--out', str(par)], check=True, capture_output=True)
    for n in ('m0', 'm3', 'm5'):
        a = json.load(open(one / 'work' / f'{n}_controls.json')); b = json.load(open(par / 'work' / f'{n}_controls.json'))
        assert a == b
    assert json.load(open(one / 'r9_controls.json'))['kill'] == json.load(open(par / 'r9_controls.json'))['kill']
