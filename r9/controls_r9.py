"""Round 9 controls: datasets built from real features by rewriting the coordinates of the subspace a score reads.

Notation (per model): C (N, k) the real subspace coordinates, mu their mean, Sigma their covariance, E_c = tr(Sigma),
G ~ N(0, Sigma) a content-free draw, u_y a unit class prototype (K, k).
  NC (negative, pure energy/offset)   C' = o * mu + sqrt(e) * G
  PC (positive, planted content)      C' = mu + sqrt(e) * (sqrt(1 - a^2) G + a sqrt(E_c) u_y)
In one dataset every model gets its own (e, o) or (e, a); the check then has to recover the cross-model order source.
"""
import numpy as np
import sanity_check_general as G

SEED = 100900
E_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
O_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
A_LEVELS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75)
A_KILL = 0.6          # PC strength at which the pre-registered power and manipulation criteria are read

STREAMS = {  # name: (stream id, subspace family, kind, prototypes, reps)
    'NC_null': (1, 'null', 'NC', None, 40),
    'PCm_null_rand': (2, 'null', 'PCm', 'rand', 40),
    'PCc_null_rand': (3, 'null', 'PCc', 'rand', 20),
    'PCm_null_text': (4, 'null', 'PCm', 'text', 20),
    'PCc_null_text': (5, 'null', 'PCc', 'text', 10),
    'NC_vim': (6, 'vim', 'NC', None, 20),
    'PCm_vim_rand': (7, 'vim', 'PCm', 'rand', 20),
    'PCc_vim_rand': (8, 'vim', 'PCc', 'rand', 10),
}
STREAM_REAL = 9      # surrogate draws on the real features
STREAM_HEAD = 10     # perturbed-head draws
BOOT_SEED = SEED


def dataset_params(stream_id, kind, level_idx, rep, n_models):
    """Per-model (e, o, a) of one control dataset; drawn once per dataset, independent of the model loop."""
    r = np.random.default_rng([SEED, stream_id, level_idx, rep])
    if kind == 'NC':
        return dict(e=r.choice(E_GRID, n_models), o=r.choice(O_GRID, n_models), a=np.zeros(n_models))
    a = A_LEVELS[level_idx] * r.permutation(np.linspace(0, 1, n_models))
    e = np.ones(n_models) if kind == 'PCm' else r.choice(E_GRID, n_models)
    return dict(e=e, o=np.ones(n_models), a=a)


def model_rng(stream_id, level_idx, rep, model_idx):
    return np.random.default_rng([SEED, stream_id, level_idx, rep, model_idx])


def random_prototypes(K, k, rng):
    U = rng.standard_normal((K, k)); return U / np.linalg.norm(U, axis=1, keepdims=True)


def text_prototypes(A_sub, T):
    """u_y proportional to A_sub^T t_y: the null direction the translator maps closest to the class text."""
    U = T @ A_sub; return U / np.linalg.norm(U, axis=1, keepdims=True)


def make_control(C, y, kind, e, o, a, U, rng, mu=None, L=None):
    """One control version of the subspace coordinates C (N, k)."""
    if mu is None:
        mu, L = G.moment_sqrt(C)
    Gd = G.gaussian_draw(len(C), mu, L, rng, centred=True)
    if kind == 'NC':
        return o * mu + np.sqrt(e) * Gd
    Ec = float(np.sum(L ** 2))
    return mu + np.sqrt(e) * (np.sqrt(1 - a ** 2) * Gd + a * np.sqrt(Ec) * U[np.asarray(y).astype(int)])
