import dataclasses
import math
import re
import torch
import sys
from collections import defaultdict
from torch import Tensor
from typing import List, Optional, Tuple
from sd_mecha import merge_method, Parameter, Return, StateDict
from sd_mecha.keys_map import KeyMapBuilder


@dataclasses.dataclass(frozen=True, slots=True)
class QkvoPolarSpec:
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class GegluRebasinPolarSpec:
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class ConcatInputPolarSpec:
    ranges: List[Tuple[int, int]]


@dataclasses.dataclass(frozen=True, slots=True)
class SimplePolarSpec:
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class MultiplicativeSpec:
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class LinearSpec:
    pass


@merge_method
class polar_121:
    clip_l_attn_re = re.compile(r"layers\.(\d+)\.self_attn\.(.+)")
    clip_g_attn_re = re.compile(r"resblocks\.(\d+)\.attn\.(.+)")
    vae_attn_re = re.compile(r"\.([a-z]+)\.mid\.attn_1\.(.+)")
    unet_attn_re = re.compile(r"\.([a-z]+)_blocks?\.(\d+\.)(?:\d+\.)?transformer_blocks\.(\d+)\.attn([12])\.(.+)")
    unet_geglu_re = re.compile(r"\.([a-z]+)_blocks?\.(\d+\.)(?:\d+\.)?transformer_blocks\.(\d+)\.ff\.net\.(.+)")

    @classmethod
    def map_keys(cls, b: KeyMapBuilder):
        attention_keys = defaultdict(list)
        geglu_keys = defaultdict(list)
        concat_input_polar_specs = {}
        simple_polar_keys = []
        log_keys = []
        linear_keys = []
        for key, meta in b.keys.items():
            if is_vae(key):
                linear_keys.append(key)
            elif (match := cls.clip_l_attn_re.search(key)) and match.group(2) != "out_proj.bias":
                layer_id = ("clip_l", match.group(1))
                attention_keys[layer_id].append(key)
            elif (match := cls.clip_g_attn_re.search(key)) and match.group(2) != "out_proj.bias":
                layer_id = ("clip_g", match.group(1))
                attention_keys[layer_id].append(key)
            # elif (match := cls.vae_attn_re.search(key)) and match.group(2) != "proj_out.bias":
            #     layer_id = ("vae", *match.group(1, 2))
            #     attention_keys[layer_id].append(key)
            elif (match := cls.unet_attn_re.search(key)) and match.group(5) != "to_out.0.bias":
                layer_id = ("unet", *match.group(1, 2, 3, 4))
                attention_keys[layer_id].append(key)
            elif (match := cls.unet_geglu_re.search(key)) and match.group(4) != "2.bias":
                layer_id = ("unet", *match.group(1, 2, 3))
                geglu_keys[layer_id].append(key)
            elif ".output_blocks." in key and key.endswith(("in_layers.2.weight", "skip_connection.weight")):
                ranges = []
                for i in range(2):
                    if meta.shape[0] * 2 >= meta.shape[1]:
                        t_start = meta.shape[0] * i
                        t_end = meta.shape[0] * (i + 1)
                    else:
                        t_start = meta.shape[0] * 2 * i
                        t_end = meta.shape[0] * 2 * (i + 1)
                    ranges.append((t_start, t_end))
                concat_input_polar_specs[key] = ConcatInputPolarSpec(ranges)
            elif key.endswith("weight") and len(meta.shape) >= 2 and "embed" not in key and ".out." not in key:
                simple_polar_keys.append(key)
            elif key.endswith("weight") and len(meta.shape) <= 1:
                log_keys.append(key)
            else:
                linear_keys.append(key)

        for group in attention_keys.values():
            b[group] = b.keys[group] @ QkvoPolarSpec()

        for group in geglu_keys.values():
            b[group] = b.keys[group] @ GegluRebasinPolarSpec()

        for key, spec in concat_input_polar_specs.items():
            b[key] = b.keys[key] @ spec

        for key in simple_polar_keys:
            b[key] = b.keys[key] @ SimplePolarSpec()

        for key in log_keys:
            b[key] = b.keys[key] @ MultiplicativeSpec()

        for key in linear_keys:
            b[key] = b.keys[key] @ LinearSpec()

    def __call__(
        self,
        a: Parameter(StateDict[Tensor], model_config="sdxl-sgm"),
        b: Parameter(StateDict[Tensor], model_config="sdxl-sgm"),
        alpha: Parameter(float, model_config="sdxl-sgm") = 0.5,
        **kwargs,
    ) -> Return(Tensor, model_config="sdxl-sgm"):
        keys = kwargs["key_relation"].outputs
        spec = kwargs["key_relation"].meta

        if math.isclose(alpha, 0.0): return {key: a[key] for key in keys}
        if math.isclose(alpha, 1.0): return {key: b[key] for key in keys}

        if isinstance(spec, LinearSpec):
            res = {key: torch.lerp(a[key], b[key], alpha) for key in keys}
        elif isinstance(spec, MultiplicativeSpec):
            res = {key: interpolate_geometric(a[key], b[key], alpha) for key in keys}
        elif isinstance(spec, QkvoPolarSpec):
            q_a, k_a, v_a, o_a = extract_qkvo_sdxl_sgm(a, keys)
            q_b, k_b, v_b, o_b = extract_qkvo_sdxl_sgm(b, keys)

            qh, s_qk, k = interpolate_polar(q_a.mH @ k_a, q_b.mH @ k_b, rank=64, alpha=alpha, keys=keys)
            del q_a, k_a, q_b, k_b
            s_qk_half = matrix_sqrt(s_qk)
            q = s_qk_half @ qh.mH
            k = s_qk_half @ k
            del qh, s_qk, s_qk_half

            vh, s_vo, o = interpolate_polar(v_a.mH @ o_a, v_b.mH @ o_b, rank=64, alpha=alpha, keys=keys)
            del v_a, o_a, v_b, o_b
            s_vo_half = matrix_sqrt(s_vo)
            v = s_vo_half @ vh.mH
            o = s_vo_half @ o
            del vh, s_vo, s_vo_half

            res = return_qkvo_sdxl_sgm(q, k, v, o, keys)
        elif isinstance(spec, GegluRebasinPolarSpec):
            k_a, g_a, o_a = extract_kgo_sdxl_sgm(a, keys)
            k_b, g_b, o_b = extract_kgo_sdxl_sgm(b, keys)

            k_b, g_b, o_b = align_geglu(
                k_a, g_a, o_a,
                k_b, g_b, o_b,
            )
            u, s, vh = interpolate_polar(k_a[None], k_b[None], rank=min(k_a.shape[-2:]), alpha=alpha, keys=keys)
            k = (u @ s @ vh).squeeze(0)
            del u, s, vh

            u, s, vh = interpolate_polar(g_a[None], g_b[None], rank=min(g_a.shape[-2:]), alpha=alpha, keys=keys)
            g = (u @ s @ vh).squeeze(0)
            del u, s, vh

            u, s, vh = interpolate_polar(o_a[None], o_b[None], rank=min(o_a.shape[-2:]), alpha=alpha, keys=keys)
            o = (u @ s @ vh).squeeze(0)
            del u, s, vh

            res = return_kgo_sdxl_sgm(k, g, o, keys)
        elif isinstance(spec, ConcatInputPolarSpec):
            res = {}
            for key in keys:
                res_parts = []
                for r in spec.ranges:
                    t_a = a[key][:, r[0]:r[1]]
                    t_b = b[key][:, r[0]:r[1]]

                    t_a_2d = t_a.flatten(start_dim=1)[None]
                    t_b_2d = t_b.flatten(start_dim=1)[None]

                    u, s, vh = interpolate_polar(t_a_2d, t_b_2d, rank=min(t_a_2d.shape[-2:]), alpha=alpha, keys=keys)
                    res_parts.append((u @ s @ vh).reshape_as(t_a))
                res[key] = torch.cat(res_parts, dim=1)
        elif isinstance(spec, SimplePolarSpec):
            res = {}
            for key in keys:
                t_a = a[key]
                t_b = b[key]
                original_shape = t_a.shape
                t_a_2d = t_a.flatten(start_dim=1)[None]
                t_b_2d = t_b.flatten(start_dim=1)[None]
                u, s, vh = interpolate_polar(t_a_2d, t_b_2d, rank=min(t_a_2d.shape[-2:]), alpha=alpha, keys=keys)
                res[key] = (u @ s @ vh).reshape(original_shape)
        else:
            raise KeyError(keys)

        return res


def interpolate_geometric(a: Tensor, b: Tensor, alpha: float) -> Tensor:
    return torch.lerp((a + 0j).log(), (b + 0j).log(), alpha).exp().real


def interpolate_polar(
    a: Tensor,
    b: Tensor,
    rank: int,
    alpha: float,
    keys: Tuple[str, ...],
) -> Tuple[Tensor, Tensor, Tensor]:
    shape_2d = a.shape[-2:]
    batch_size = a.shape[0]
    device = a.device
    dtype = a.dtype

    u_a, s_a, vh_a = svd(a, rank)
    u_b, s_b, vh_b = svd(b, rank)
    s_a = s_a.unsqueeze(-2)
    s_b = s_b.unsqueeze(-2)

    if rank < min(shape_2d):
        flip_v = (vh_b @ vh_a.mH).slogdet()[0] < 0
        flip_u = (u_b.mH @ u_a).slogdet()[0] < 0
        vh_b[flip_v, ..., -1, :] *= -1
        u_b[flip_u, ..., -1] *= -1

        q = polar_part((u_a * s_a).mH @ (u_b * s_b) + (vh_a * s_a.mH) @ (vh_b * s_b.mH).mH)
        u_b = u_b @ q.mH
        vh_b = q @ vh_b

        u = interpolate_stiefel(u_a, u_b, alpha, keys, 1e-6, 100)
        vh = interpolate_stiefel(vh_a.mH, vh_b.mH, alpha, keys, 1e-6, 100).mH
    elif shape_2d[-2] > shape_2d[-1]:
        flip_v = (vh_b @ vh_a.mH).slogdet()[0] < 0
        vh_b[flip_v, ..., -1, :] *= -1
        u_b[flip_v, ..., -1] *= -1

        q = polar_part((u_a * s_a).mH @ (u_b * s_b) + (vh_a * s_a.mH) @ (vh_b * s_b.mH).mH)
        u_b = u_b @ q.mH
        vh_b = q @ vh_b

        u = interpolate_stiefel(u_a, u_b, alpha, keys, 1e-6, 100)
        vh = interpolate_orthogonal(vh_a.mH, vh_b.mH, alpha).mH
    elif shape_2d[-2] < shape_2d[-1]:
        flip_u = (u_b.mH @ u_a).slogdet()[0] < 0
        vh_b[flip_u, ..., -1, :] *= -1
        u_b[flip_u, ..., -1] *= -1

        q = polar_part((u_a * s_a).mH @ (u_b * s_b) + (vh_a * s_a.mH) @ (vh_b * s_b.mH).mH)
        u_b = u_b @ q.mH
        vh_b = q @ vh_b

        u = interpolate_orthogonal(u_a, u_b, alpha)
        vh = interpolate_stiefel(vh_a.mH, vh_b.mH, alpha, keys, 1e-6, 100).mH
    else:
        flip_v = (vh_b @ vh_a.mH).slogdet()[0] < 0
        flip_u = (u_b.mH @ u_a).slogdet()[0] < 0
        vh_b[flip_v, ..., -1, :] *= -1
        u_b[flip_u, ..., -1] *= -1

        q = polar_part((u_a * s_a).mH @ (u_b * s_b) + (vh_a * s_a.mH) @ (vh_b * s_b.mH).mH)
        u_b = u_b @ q.mH
        vh_b = q @ vh_b

        u = interpolate_orthogonal(u_a, u_b, alpha)
        vh = interpolate_orthogonal(vh_a.mH, vh_b.mH, alpha).mH

    eye = torch.eye(rank, device=device, dtype=dtype)[None].repeat(batch_size, 1, 1)
    s = interpolate_psd_square(eye, s_a, q, s_b, alpha, keys)
    return u, s, vh


def matrix_sqrt(m):
    v, vs = torch.linalg.eigh(m)
    v_half = v.unsqueeze(-2)**0.5
    res = (vs * v_half) @ vs.mH
    return res


def polar_part(a, cancel_reflection=False):
    u, _, vh = svd(a, cancel_reflection=cancel_reflection)
    return u @ vh


def interpolate_stiefel(a, b, t, keys, tau=None, max_iter=100):
    delta = log_stiefel(a, b, keys, tau, max_iter)
    res = exp_stiefel(a, t * delta)
    return res


def exp_stiefel(U, Delta):
    A = U.mH @ Delta
    Y = Delta - U @ A
    Q, R = qr_pos(Y)
    W = torch.cat((torch.cat((A, -R.mH), -1), torch.cat((R, torch.zeros_like(A)), -1)), -2)
    M = torch.linalg.matrix_exp(W)
    p = A.shape[-2]
    return U @ M[..., :p, :p] + Q @ M[..., p:, :p]


def log_stiefel(a, b, keys, tau=None, max_iter=100):
    original_shape = a.shape
    while len(a.shape) < 3:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    a = a.flatten(end_dim=-3)
    b = b.flatten(end_dim=-3)
    batch_size, n, p = b.shape
    assert n > p
    k = min(n-p, p)
    tau = tau or 1e-8
    assert max_iter >= 1

    m = a.mH @ b

    q, n_mat = qr_pos(b - a @ m)
    q = q[..., :k]
    n_mat = n_mat[..., :k, :]
    v = orthogonal_complete(torch.cat((m, n_mat), dim=-2))

    r, sigma, r_hat_t = svd(v[..., p:, p:], k)
    q @= r
    v[..., p:, :p] = r.mH @ n_mat
    v[..., :p, p:] @= r_hat_t.mH
    p_arange = torch.arange(p, p+k, device=v.device)
    v[..., p:, p:].zero_()
    v[..., p_arange, p_arange] = sigma
    del r, sigma, r_hat_t, p_arange

    v[v.slogdet()[0] < 0, ..., -1] *= -1

    k_arange = torch.arange(k, device=v.device, dtype=torch.long)
    printed_error = False
    l = None
    for i in range(max_iter):
        l = canonical_ort_log(v)
        c = l[..., p:, p:]
        c_norm_idx = torch.linalg.matrix_norm(c).argmax()
        c_norm = torch.linalg.matrix_norm(c[c_norm_idx])
        if c_norm > 10:
            print(f"warning: log error very high {c_norm.item():0.3f} at iteration {i}, batch {c_norm_idx}, key {keys[0]}", file=sys.stderr)
            printed_error = True
        elif printed_error:
            print(f"warning: log error started converging {c_norm.item():0.3f} at iteration {i}, batch {c_norm_idx}, key {keys[0]}", file=sys.stderr)
            printed_error = False
        if c_norm <= tau:
            # print(f"stiefel error: {c_norm.item()} at iteration {i}, batch {c_norm_idx}, key {kwargs['key']}", file=sys.stderr)
            break
        elif i == max_iter - 1:
            print(f"stiefel error: {c_norm.item()}, batch {c_norm_idx}, key {keys[0]}", file=sys.stderr)

        s = l[..., p:, :p] @ l[..., p:, :p].mH / 12
        s[..., k_arange, k_arange] -= 0.5
        g = solve_symmetric_sylvester(s, c)
        v[..., p:] @= torch.linalg.matrix_exp(g)

    delta = a @ l[..., :p, :p] + q @ l[..., p:, :p]
    return delta.reshape(original_shape)


def orthogonal_complete(q: torch.Tensor) -> torch.Tensor:
    batch_size, n, k = q.shape
    if n <= k:
        return q

    m = torch.eye(n, device=q.device, dtype=q.dtype)[:, k:]
    p = m - q @ (q.mH @ m)
    q2 = qr_pos(p)[0]
    return torch.cat([q, q2], dim=-1)


def qr_pos(x):
    q, r = torch.linalg.qr(x)
    s = torch.sign(r.diagonal(offset=0, dim1=-2, dim2=-1))
    s[s == 0] = 1
    return q * s.unsqueeze(-2), r / s.unsqueeze(-1)


def solve_symmetric_sylvester(s, c):
    v, vs = torch.linalg.eigh(s)
    c_t = vs.mH @ c @ vs
    d = v.unsqueeze(-2) + v.unsqueeze(-1)
    if torch.any(torch.abs(d) < 1e-12):
        print("Singular Sylvester operator: some λ_i+λ_j ≈ 0", file=sys.stderr)

    g_t = c_t / d
    g = vs @ g_t @ vs.mH
    return g


def interpolate_psd_square(u_a, s_a, u_b, s_b, t, keys):
    p_a_sqrt = (u_a * s_a.sqrt()) @ u_a.mH
    p_a_inv_sqrt = (u_a * (s_a.rsqrt()).nan_to_num(0)) @ u_a.mH
    p_b = (u_b * s_b) @ u_b.mH

    h, hs = torch.linalg.eigh(p_a_inv_sqrt @ p_b @ p_a_inv_sqrt.mH)
    h = h.reshape_as(s_a)
    p = (p_a_sqrt+0j) @ ((hs+0j) * (h+0j)**t) @ (hs.mH @ p_a_sqrt.mH + 0j)
    if p.imag.abs().max() > 1e-4:
        print(f"imaginary component in p: abs max {p.imag.abs().max()}, key {keys[0]}")

    return p.real


def interpolate_orthogonal(q_a, q_b, alpha):
    return q_a @ torch.linalg.matrix_exp(alpha * canonical_ort_log(q_a.mH @ q_b))


def canonical_ort_log(Q: torch.Tensor, tol=1e-8):
    """
    Real, skew-symmetric logarithm for Q in SO(n).
    Handles even multiplicity of -1 by explicitly building π-planes.
    Returns a real tensor (no complex dtype).
    """
    assert Q.dim() >= 2
    n = Q.shape[-1]
    device, dtype = Q.device, Q.dtype
    I = torch.eye(n, device=device, dtype=dtype)
    batch = Q.reshape(-1, n, n)
    B = batch.shape[0]
    A = torch.zeros_like(batch)

    # Build basis for (-1)-eigenspace via SVD of Q+I
    U, S, Vh = torch.linalg.svd(batch + I)
    # multiplicity per batch item
    k = (S < tol).sum(dim=1)  # (B,)

    for b in range(B):
        kb = int(k[b].item())
        if kb:
            # Use last kb right-singular vectors as nullspace basis
            E = Vh[b, -kb:].T  # (n, kb)
            # Orthonormalize columns (QR)
            Qe, _ = torch.linalg.qr(E)  # (n, kb)
            # Pair columns into 2D planes
            assert kb % 2 == 0, "For SO(n) the -1 multiplicity should be even."
            for i in range(0, kb, 2):
                u = Qe[:, i]
                v = Qe[:, i + 1]
                A[b] += math.pi * (u[:, None] @ v[None, :] - v[:, None] @ u[None, :])

    # Complement: principal skew log (no -1 eigenvalues on that subspace).
    # Use complex eig, then project to skew-Hermitian and take real part.
    W, V = torch.linalg.eig(batch)  # (B,n), (B,n,n)
    theta = torch.angle(W)  # principal angles in (-pi, pi]
    Scomp = torch.linalg.solve(V, V * (1j * theta).unsqueeze(-2), left=False)
    Scomp = 0.5 * (Scomp - Scomp.mH)
    A = A + Scomp.real.to(dtype)

    A = 0.5 * (A - A.mH)
    return A.reshape(Q.shape)


def svd(q, rank=None, cancel_reflection: bool = False):
    if rank is None:
        rank = min(q.shape[-2:])

    svd_driver = "gesvd" if q.is_cuda else None
    if rank < min(q.shape[-2:]):
        u, s, v = torch_svd_lowrank(q, driver=svd_driver, q=rank)
        vh = v.mH
        del v
    else:
        u, s, vh = torch.linalg.svd(q, full_matrices=False, driver=svd_driver)
        if cancel_reflection and min(q.shape[-2:]) == max(q.shape[-2:]):
            u[..., -1] /= (torch.slogdet(u)[0] * torch.slogdet(vh)[0]).unsqueeze(-1)

    return u, s, vh


class MatmulIdentity:
    def __matmul__(self, other):
        return other

    def __rmatmul__(self, other):
        return other

    @property
    def mH(self):
        return self

    @property
    def mT(self):
        return self

    @property
    def H(self):
        return self

    @property
    def T(self):
        return self

    def to(self, *args, **kwargs):
        return self


# need to redefine torch.svd_lowrank to specify the svd driver
def torch_svd_lowrank(
    A: Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
    driver: Optional[str] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    q = 6 if q is None else q
    m, n = A.shape[-2:]
    if M is None:
        M_t = None
    else:
        M_t = transpose(M)
    A_t = transpose(A)

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n or n > q:
        # computing the SVD approximation of a transpose in
        # order to keep B shape minimal (the m < n case) or the V
        # shape small (the n > q case)
        Q = get_approximate_basis(A_t, q, niter=niter, M=M_t)
        Q_c = conjugate(Q)
        if M is None:
            B_t = matmul(A, Q_c)
        else:
            B_t = matmul(A, Q_c) - matmul(M, Q_c)
        assert B_t.shape[-2] == m, (B_t.shape, m)
        assert B_t.shape[-1] == q, (B_t.shape, q)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, driver=driver, full_matrices=False)
        V = Vh.mH
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter, M=M)
        Q_c = conjugate(Q)
        if M is None:
            B = matmul(A_t, Q_c)
        else:
            B = matmul(A_t, Q_c) - matmul(M_t, Q_c)
        B_t = transpose(B)
        assert B_t.shape[-2] == q, (B_t.shape, q)
        assert B_t.shape[-1] == n, (B_t.shape, n)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, driver=driver, full_matrices=False)
        V = Vh.mH
        U = Q.matmul(U)

    return U, S, V


def get_approximate_basis(A: Tensor, q: int, niter: Optional[int] = 0, M: Optional[Tensor] = None) -> Tensor:
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    niter = 2 if niter is None else niter
    m, n = A.shape[-2:]
    dtype = A.dtype

    R = torch.eye(n, q, dtype=dtype, device=A.device)

    # The following code could be made faster using torch.geqrf + torch.ormqr
    # but geqrf is not differentiable
    A_H = transjugate(A)
    if M is None:
        Q = torch.linalg.qr(matmul(A, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q)).Q
    else:
        M_H = transjugate(M)
        Q = torch.linalg.qr(matmul(A, R) - matmul(M, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q) - matmul(M_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q) - matmul(M, Q)).Q

    return Q


def transjugate(A):
    """Return transpose conjugate of a matrix or batches of matrices."""
    return conjugate(transpose(A))


def conjugate(A):
    """Return conjugate of tensor A.

    .. note:: If A's dtype is not complex, A is returned.
    """
    if A.is_complex():
        return A.conj()
    return A


def transpose(A):
    """Return transpose of a matrix or batches of matrices."""
    ndim = len(A.shape)
    return A.transpose(ndim - 1, ndim - 2)


def matmul(A: Optional[Tensor], B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    if A is None:
        return B
    if is_sparse(A):
        return torch.sparse.mm(A, B)
    return torch.matmul(A, B)


def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    if isinstance(A, torch.Tensor):
        return A.layout == torch.sparse_coo

    error_str = "expected Tensor"
    if not torch.jit.is_scripting():
        error_str += f" but got {type(A)}"
    raise TypeError(error_str)


def extract_kgo_sdxl_sgm(sd: StateDict[Tensor], keys: Tuple[str, ...]) -> Tuple[Tensor, Tensor, Tensor]:
    if len(keys) == 1:
        raise RuntimeError(f"unknown keys: {keys}")

    if is_unet(keys[0]):
        (
            key_kg_bias, key_kg,
            key_o,
        ) = keys

        fc1_bias, fc1 = sd[key_kg_bias], sd[key_kg]
        o = sd[key_o]

        k, g = bundle_weight_bias(fc1, fc1_bias).chunk(2)
        o = o.mT
    else:
        raise RuntimeError(f"unknown keys: {keys}")

    return k, g, o


def return_kgo_sdxl_sgm(k, g, o, keys):
    if len(keys) == 1:
        raise RuntimeError(f"unknown keys: {keys}")

    if is_unet(keys[0]):
        (
            key_kg_bias, key_kg,
            key_o,
        ) = keys

        kg = torch.cat((k, g), dim=0)
        kg, kg_bias = split_weight_bias(kg)
        o = o.mT

        result = {
            key_kg_bias: kg_bias, key_kg: kg,
            key_o: o
        }
    else:
        raise RuntimeError(f"unknown keys: {keys}")

    return result


def extract_qkvo_sdxl_sgm(sd: StateDict[Tensor], keys: Tuple[str, ...]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if len(keys) == 1:
        raise RuntimeError(f"bad keys: {keys}")

    if is_clip_l(keys[0]) or is_vae(keys[0]):
        (
            key_k_bias, key_k,
            key_o,
            key_q_bias, key_q,
            key_v_bias, key_v,
        ) = keys

        k_bias, k = sd[key_k_bias], sd[key_k]
        o = sd[key_o]
        q_bias, q = sd[key_q_bias], sd[key_q]
        v_bias, v = sd[key_v_bias], sd[key_v]

        k = bundle_weight_bias(k, k_bias).unflatten(0, (-1, 64))
        q = bundle_weight_bias(q, q_bias).unflatten(0, (-1, 64))
        v = bundle_weight_bias(v, v_bias).unflatten(0, (-1, 64))
        o = o.mT.unflatten(0, (-1, 64))
    elif is_clip_g(keys[0]):
        (
            key_bias, key_weight,
            key_o,
        ) = keys

        bias = sd[key_bias]
        weight = sd[key_weight]
        o = sd[key_o]

        dim = weight.shape[0] // 3

        q = bundle_weight_bias(weight[:dim], bias[:dim]).unflatten(0, (-1, 64))
        k = bundle_weight_bias(weight[dim:dim*2], bias[dim:dim*2]).unflatten(0, (-1, 64))
        v = bundle_weight_bias(weight[dim*2:], bias[dim*2:]).unflatten(0, (-1, 64))
        o = o.mT.unflatten(0, (-1, 64))
    elif is_unet(keys[0]):
        key_k, key_o, key_q, key_v = keys

        k = sd[key_k].unflatten(0, (-1, 64))
        o = sd[key_o].mT.unflatten(0, (-1, 64))
        q = sd[key_q].unflatten(0, (-1, 64))
        v = sd[key_v].unflatten(0, (-1, 64))
    else:
        raise KeyError(keys)

    return q, k, v, o


def return_qkvo_sdxl_sgm(q, k, v, o, keys):
    if is_clip_l(keys[0]) or is_vae(keys[0]):
        (
            key_k_bias, key_k,
            key_o,
            key_q_bias, key_q,
            key_v_bias, key_v,
        ) = keys

        k, k_bias = split_weight_bias(k.flatten(end_dim=1))
        o = o.flatten(end_dim=1).mT
        q, q_bias = split_weight_bias(q.flatten(end_dim=1))
        v, v_bias = split_weight_bias(v.flatten(end_dim=1))

        result = {
            key_k_bias: k_bias, key_k: k,
            key_o: o,
            key_q_bias: q_bias, key_q: q,
            key_v_bias: v_bias, key_v: v,
        }
    elif is_clip_g(keys[0]):
        (
            key_bias, key_weight,
            key_o,
        ) = keys

        k, k_bias = split_weight_bias(k.flatten(end_dim=1))
        o = o.flatten(end_dim=1).mT
        q, q_bias = split_weight_bias(q.flatten(end_dim=1))
        v, v_bias = split_weight_bias(v.flatten(end_dim=1))

        weight = torch.cat((q, k, v), dim=0)
        bias = torch.cat((q_bias, k_bias, v_bias), dim=0)

        result = {
            key_bias: bias, key_weight: weight,
            key_o: o,
        }
    elif is_unet(keys[0]):
        (
            key_k,
            key_o,
            key_q,
            key_v,
        ) = keys

        k = k.flatten(end_dim=1)
        o = o.flatten(end_dim=1).mT
        q = q.flatten(end_dim=1)
        v = v.flatten(end_dim=1)

        result = {
            key_k: k,
            key_o: o,
            key_q: q,
            key_v: v,
        }
    else:
        raise KeyError(keys)

    return result


def bundle_weight_bias(w, b):
    b = b.unsqueeze(-1)
    wb = torch.cat([w, b], dim=-1)
    return wb


def split_weight_bias(wb):
    b = wb[..., -1]
    w = wb[..., :-1]
    return w, b


def is_clip_l(k):
    return k.startswith("conditioner.embedders.0")


def is_clip_g(k):
    return k.startswith("conditioner.embedders.1")


def is_vae(k):
    return k.startswith("first_stage_model")


def is_unet(k):
    return k.startswith("model.diffusion_model")


import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def align_geglu(
    WkA: torch.Tensor,  # (m, d_k)
    WgA: torch.Tensor,  # (m, d_g)
    WoA: torch.Tensor,  # (m, d_out)
    WkB: torch.Tensor,  # (m, d_k)
    WgB: torch.Tensor,  # (m, d_g)
    WoB: torch.Tensor,  # (m, d_out)
    block_cols: int = 256,
    bisect_iters: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      perm:  (m,) int64, mapping A neuron i -> B neuron perm[i]
      alpha: (m,) same dtype/device as inputs, signed scale applied to B neuron perm[i]

    Objective per matched (i,j) with alpha = p*exp(z), p in {+1,-1}:
      ||WkA[i] - alpha WkB[j]||^2
    + ||WoA[:,i] - alpha^{-1} WoB[:,j]||^2
    + ||WgA[i] - WgB[j]||^2
    """
    device = WkA.device
    dtype = WkA.dtype

    m, dk = WkA.shape
    assert WkB.shape == (m, dk)
    assert WgA.shape[0] == m and WgB.shape[0] == m
    m2, dout = WoA.shape
    # assert m2 == m and WoB.shape == (dout, m)

    # Precompute norms in float64
    WkA64, WkB64 = WkA.double(), WkB.double()
    WgA64, WgB64 = WgA.double(), WgB.double()
    WoA64, WoB64 = WoA.double(), WoB.double()

    ak2 = (WkA64 * WkA64).sum(dim=1)          # (m,)
    ao2 = (WoA64 * WoA64).sum(dim=1)          # (m,)
    ag2 = (WgA64 * WgA64).sum(dim=1)          # (m,)

    bk2 = (WkB64 * WkB64).sum(dim=1)          # (m,)
    bo2 = (WoB64 * WoB64).sum(dim=1)          # (m,)
    bg2 = (WgB64 * WgB64).sum(dim=1)          # (m,)

    # Output buffers
    pair_cost = torch.empty((m, m), device=device, dtype=torch.float64)
    best_alpha = torch.empty((m, m), device=device, dtype=torch.float64)

    eps = 1e-12

    def solve_alpha_logspace_block(Acoef, Bcoef, Cdot, Ddot, ak2_blk, ao2_blk):
        # All args: float64, shape (m, nb)
        # Returns alpha*, cost_part* (k/o terms only), both float64, shape (m, nb)

        alpha_best = torch.ones_like(Acoef)
        cost_best = torch.full_like(Acoef, float("inf"))

        A0 = Acoef.abs() <= eps
        B0 = Bcoef.abs() <= eps

        # A==0,B==0: constant
        mask = A0 & B0
        if mask.any():
            cost_best[mask] = (ak2_blk + ao2_blk)[mask]
            alpha_best[mask] = 1.0

        # A==0,B>0: minimize over t=alpha^{-1}
        mask = A0 & (~B0)
        if mask.any():
            t = Ddot[mask] / Bcoef[mask]
            t = torch.where(t.abs() > eps, t, torch.sign(t) * eps + (t == 0) * eps)
            alpha = 1.0 / t
            inv_alpha = t
            cost = ak2_blk[mask] + (ao2_blk[mask] + (inv_alpha ** 2) * Bcoef[mask] - 2.0 * inv_alpha * Ddot[mask])
            alpha_best[mask] = alpha
            cost_best[mask] = cost

        # B==0,A>0: minimize linear least squares in alpha
        mask = (~A0) & B0
        if mask.any():
            alpha = Cdot[mask] / Acoef[mask]
            cost = (ak2_blk[mask] + (alpha ** 2) * Acoef[mask] - 2.0 * alpha * Cdot[mask]) + ao2_blk[mask]
            alpha_best[mask] = alpha
            cost_best[mask] = cost

        # General case: A>0,B>0
        mask = (~A0) & (~B0)
        if not mask.any():
            return alpha_best, cost_best

        # Pull masked coefficients ONCE
        A_m = Acoef[mask]
        B_m = Bcoef[mask]
        C_m = Cdot[mask]
        D_m = Ddot[mask]
        ak2_m = ak2_blk[mask]
        ao2_m = ao2_blk[mask]

        def f_and_g_masked(z, p):
            ez = torch.exp(z)
            emz = torch.exp(-z)
            e2z = ez * ez
            e_2z = emz * emz
            g = A_m * e2z - (p * C_m) * ez - B_m * e_2z + (p * D_m) * emz
            f = (ak2_m + A_m * e2z - 2.0 * (p * C_m) * ez
                 + ao2_m + B_m * e_2z - 2.0 * (p * D_m) * emz)
            return f, g

        def solve_for_p(p):
            # Deterministic bracket radius (uses abs of masked dot products)
            t = torch.maximum(torch.log1p(C_m.abs() / A_m), torch.log1p(D_m.abs() / B_m)) + 2.0
            zL = -t
            zR = +t
            _, gL = f_and_g_masked(zL, p)
            _, gR = f_and_g_masked(zR, p)

            # Optional bounded expansion for non-finite values
            for _ in range(3):
                finite = torch.isfinite(gL) & torch.isfinite(gR)
                br = finite & (((gL <= 0) & (gR >= 0)) | ((gL >= 0) & (gR <= 0)))
                if br.all():
                    break
                t = torch.where(br, t, t * 2.0)
                zL = -t
                zR = +t
                _, gL = f_and_g_masked(zL, p)
                _, gR = f_and_g_masked(zR, p)

            for _ in range(bisect_iters):
                zM = 0.5 * (zL + zR)
                _, gM = f_and_g_masked(zM, p)
                same_as_L = (gM >= 0) == (gL >= 0)
                zL = torch.where(same_as_L, zM, zL)
                gL = torch.where(same_as_L, gM, gL)
                zR = torch.where(same_as_L, zR, zM)
                gR = torch.where(same_as_L, gR, gM)

            z_star = 0.5 * (zL + zR)
            f_star, _ = f_and_g_masked(z_star, p)
            alpha_star = p * torch.exp(z_star)
            return alpha_star, f_star

        a_pos, f_pos = solve_for_p(+1.0)
        a_neg, f_neg = solve_for_p(-1.0)

        take_pos = f_pos <= f_neg
        a_star = torch.where(take_pos, a_pos, a_neg)
        f_star = torch.where(take_pos, f_pos, f_neg)

        # scatter back into 2D outputs
        idx = mask.nonzero(as_tuple=True)
        alpha_best[idx] = a_star
        cost_best[idx] = f_star
        return alpha_best, cost_best

    for j0 in range(0, m, block_cols):
        j1 = min(m, j0 + block_cols)
        nb = j1 - j0

        # Dots for this block
        Cdot = WkA64 @ WkB64[j0:j1].T              # (m, nb)
        Ddot = WoA64 @ WoB64[j0:j1].T              # (m, nb)
        Gdot = WgA64 @ WgB64[j0:j1].T              # (m, nb)

        Acoef = bk2[j0:j1].unsqueeze(0).expand(m, nb)
        Bcoef = bo2[j0:j1].unsqueeze(0).expand(m, nb)
        ak2_blk = ak2.unsqueeze(1).expand(m, nb)
        ao2_blk = ao2.unsqueeze(1).expand(m, nb)

        alpha_blk, cost_ko = solve_alpha_logspace_block(Acoef, Bcoef, Cdot, Ddot, ak2_blk, ao2_blk)

        # Add g matching cost: ||ag-bg||^2 = ||ag||^2 + ||bg||^2 - 2<ag,bg>
        g_cost = (
            ag2.unsqueeze(1).expand(m, nb)
            + bg2[j0:j1].unsqueeze(0).expand(m, nb)
            - 2.0 * Gdot
        )

        pair_cost[:, j0:j1] = cost_ko + g_cost
        best_alpha[:, j0:j1] = alpha_blk

    # Hungarian (CPU SciPy)
    cost_cpu = pair_cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_cpu)
    # row_ind is 0..m-1
    perm = torch.from_numpy(col_ind.astype(np.int64)).to(device=device)
    alpha = best_alpha[torch.arange(m, device=device), perm].to(device=device, dtype=dtype)

    return apply_geglu_perm_alpha(WkB, WgB, WoB, perm, alpha)


@torch.no_grad()
def apply_geglu_perm_alpha(
    WkB: torch.Tensor,  # (m, d_k)
    WgB: torch.Tensor,  # (m, d_g)
    WoB: torch.Tensor,  # (m, d_out)
    perm: torch.Tensor, # (m,) int64, A i -> B perm[i]
    alpha: torch.Tensor # (m,) float, signed scales for matched B neurons (in A order)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns aligned copies (WkB_new, WgB_new, WoB_new), where:
      - neurons are reordered so index i corresponds to A's neuron i
      - Wk is scaled by alpha[i]
      - Wo column is scaled by alpha[i]^{-1}
      - Wg is only permuted (not scaled)
    """
    dtype = WkB.dtype
    m = WkB.shape[0]
    assert perm.shape == (m,) and perm.dtype == torch.int64
    assert alpha.shape == (m,)

    # Permute: rows of Wk/Wg, columns of Wo
    WkP = WkB.index_select(0, perm)
    WgP = WgB.index_select(0, perm)
    WoP = WoB.index_select(0, perm)

    # Scale: Wk *= alpha, Wo *= alpha^{-1}
    eps = torch.finfo(dtype).tiny
    inv_alpha = torch.where(alpha.abs() > eps, 1.0 / alpha, torch.sign(alpha) / eps)

    Wk_new = WkP * alpha.unsqueeze(1)
    Wo_new = WoP * inv_alpha.unsqueeze(1)
    Wg_new = WgP  # not scaled

    return Wk_new, Wg_new, Wo_new


import sd_mecha

a = sd_mecha.model(r"F:\sd\models\Stable-diffusion\noobaiXLNAIXL_epsilonPred11Version.safetensors")
b = sd_mecha.model(r"F:\sd\models\Stable-diffusion\animagine-xl-4.0-zero.safetensors")

sd_mecha.merge(polar_121(a, b), merge_device="cuda:0", merge_dtype=torch.float64, threads=0, output=r"F:\sd\models\Stable-diffusion\polar_121.safetensors")
