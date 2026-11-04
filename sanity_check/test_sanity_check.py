"""The sanity-check wrapper reproduces the committed post hoc results of the paper exactly.

Fast test (committed aggregates only): rank mode on r8/posthoc/rho_decomp.json and offset_surr.json gives the same tau and
paired CIs, to the last bit, as r8/posthoc/offset_tau.json and offset_surr_tau.json (all13, E12, S11).
Slow test (skipped unless IC014_FEATS points at the private features): features mode on ResNet50 gives the committed
rho decomposition bit for bit, and the committed AS variants to a relative 1e-9 (they differ at about 1e-11, from
floating-point summation order).
Run: python -m pytest -q sanity_check   (or python sanity_check/test_sanity_check.py)"""
import json, os, sys
import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(H); sys.path.insert(0, H)
import null_sanity_check as nsc  # noqa: E402

PH = os.path.join(R, 'r8', 'posthoc')
J = lambda f: json.load(open(os.path.join(PH, f)))


def _roundtrip(x):
    return json.loads(json.dumps(x, default=float))


def _per_model():
    ms, col, _ = nsc.load_models(); D = J('rho_decomp.json'); P = J('offset_surr.json'); AS = dict(zip(ms, col('AS_SING')))
    return {m: dict(rho=D[m]['rho_table'], rho_centred=D[m]['rho_between'] + D[m]['rho_within'], rho_offset=D[m]['rho_offset'],
                    rho_between=D[m]['rho_between'], rho_within=D[m]['rho_within'], AS_SING=float(AS[m]),
                    AS=P[m]['AS'], real_c=P[m]['real_c'], off_nat=P[m]['off_nat'], zm_c=P[m]['zm_c'], perm_c=P[m]['perm_c'])
            for m in ms}


def test_rank_mode_reproduces_committed_offset_results():
    ref, set_rho = nsc.sing_reference(); PM = _per_model()
    # Step 1 (centring): offset_tau.json
    a = _roundtrip(nsc.run_rank(PM, 'AS_SING', ref, 'sing', set_rho,
                                ['AS_SING', 'rho', 'rho_centred', 'rho_offset', 'rho_between', 'rho_within']))
    T = J('offset_tau.json')
    name = {'rho': 'rho (uncentred, paper)', 'rho_centred': 'rho_centred = between+within', 'rho_offset': 'rho_offset',
            'rho_between': 'rho_between', 'rho_within': 'rho_within', 'AS_SING': 'AS_SING'}
    assert a['sets'] == T['sets']
    n = 0
    for s in T['sets']:
        for c, k in name.items():
            assert a['tau'][s][c] == T['tau'][s][k], (s, c); n += 1
        for c in ['rho', 'rho_centred', 'rho_offset', 'rho_between', 'rho_within']:
            assert a['paired'][s][f'{c} - AS_SING'] == T['paired'][s][f'{name[c]} - AS_SING'], (s, c); n += 1
    # Step 2 (matched-energy surrogates): offset_surr_tau.json
    b = _roundtrip(nsc.run_rank(PM, 'AS', ref, 'sing', set_rho, ['AS', 'real_c', 'off_nat', 'zm_c', 'perm_c']))
    U = J('offset_surr_tau.json')
    name2 = {'AS': 'AS (real, paper)', 'real_c': 'AS real centred (offset removed)', 'off_nat': 'AS offset only (natural size)',
             'zm_c': 'zm_c N(0,Sigma) @ centred rho', 'perm_c': 'perm_c (C-mu permuted) @ centred rho'}
    for s in T['sets']:
        for c, k in name2.items():
            assert b['tau'][s][c] == U['tau'][s][k], (s, c); n += 1
        assert b['paired'][s]['real_c - AS'] == U['paired'][s]['AS real centred (offset removed) - AS (real, paper)']; n += 1
    print(f'rank mode: {n} tau / paired-CI entries identical to the committed JSON')


def test_features_mode_reproduces_committed_resnet50():
    fr = os.environ.get('IC014_FEATS')
    if not fr:
        try:
            import pytest; pytest.skip('IC014_FEATS not set (private features)')
        except ImportError:
            print('skip: IC014_FEATS not set'); return
    p = next(f for f in (os.path.join(fr, e, 'ret', f'{e}_feats', 'resnet_full.npz') for e in ('e2', 'e3')) if os.path.exists(f))
    import torch
    text = nsc.DEFAULT_TEXT
    torch.set_num_threads(8)       # as r8/posthoc/rho_decomp.py
    o = nsc.run_features([p], ['resnet'])['resnet']
    D = J('rho_decomp.json')['resnet']
    for k in ['rho', 'rho_offset', 'rho_between', 'rho_within', 'rho_between_debiased', 'share_offset']:
        assert o[k] == D[k], k
    assert o['d_null'] == D['d_null']
    if os.path.exists(text):
        torch.set_num_threads(6)   # as r8/posthoc/offset_surr.py; the thread count changes float summation order
        o.update(nsc.as_variants(p, nsc.text_targets(text)))
        P = J('offset_surr.json')['resnet']
        for k in ['AS', 'off_nat', 'real_c', 'zm_c', 'perm_c', 'rho_c']:
            assert abs(o[k] - P[k]) <= 1e-9 * abs(P[k]), k   # AS: equal to ~1e-11 (BLAS summation order); rho terms above: bit-identical
    print('features mode: ResNet50 decomposition' + (' and AS variants' if 'AS' in o else '') + ' match the committed JSON (rho terms bit-identical)')


if __name__ == '__main__':
    test_rank_mode_reproduces_committed_offset_results(); test_features_mode_reproduces_committed_resnet50()
