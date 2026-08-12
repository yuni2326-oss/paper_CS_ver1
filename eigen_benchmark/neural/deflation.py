"""deflation 전략 — **이 모듈이 논문의 귀속 도구다.**

같은 구적·같은 초기화에서 모드3 성공률이 사영 100 % / 페널티 0 %로 갈린다(스파이크).
따라서 "고차모드 실패"는 spectral bias가 아니라 직교성 강제 기전의 문제다.

네 전략:
  NoDeflation          최저모드용
  Penalty(w)           v3 §3.3의 arm (a). 모드간 Rayleigh 갭(λ⁴: 12→486→3807)을
                       가중이 못 이기면 최저모드로 되돌아간다.
  ProjectionExact      arm (b). φ̃ = φ − Φ(ΦᵀWΦ)⁻¹ΦᵀWφ. v3 [2]의 충실한 대응물.
  ProjectionDiagonal   arm (a′). 파일럿 구현의 대각근사 c_k = ⟨φ,φ_k⟩/⟨φ_k,φ_k⟩ —
                       하위모드가 서로 직교할 때만 정확하고, 아니면 누출이 남는다.

**φ″도 함께 사영해야 한다.** 분자가 하위모드 곡률을 품으면 RQ가 무의미해진다.
"""
from __future__ import annotations

import torch


def project_exact(phi, Phi, wq):
    """c = (ΦᵀWΦ)⁻¹ΦᵀWφ — 완전한 그람 역행렬."""
    G = torch.einsum("ki,i,li->kl", Phi, wq, Phi)
    return torch.linalg.solve(G, torch.einsum("ki,i,i->k", Phi, wq, phi))


def project_diagonal(phi, Phi, wq):
    """c_k = ⟨φ,φ_k⟩/⟨φ_k,φ_k⟩ — 그람의 대각만 쓰는 파일럿 근사."""
    num = torch.einsum("ki,i,i->k", Phi, wq, phi)
    den = torch.einsum("ki,i,ki->k", Phi, wq, Phi)
    return num / den


class NoDeflation:
    name = "none"

    def apply(self, phi, d2, Phi, Phi2, wq):
        return phi, d2, phi.new_zeros(())


class Penalty:
    """arm (a) — 함수는 그대로 두고 겹침의 제곱을 손실에 더한다."""

    def __init__(self, weight: float):
        self.weight = float(weight)
        self.name = f"penalty(w={weight:g})"

    def apply(self, phi, d2, Phi, Phi2, wq):
        if Phi is None:
            return phi, d2, phi.new_zeros(())
        ov = torch.einsum("ki,i,i->k", Phi, wq, phi) ** 2
        nrm = (wq * phi ** 2).sum() * torch.einsum("ki,i,ki->k", Phi, wq, Phi)
        return phi, d2, self.weight * (ov / nrm).sum()


class _Projection:
    def __init__(self, coeff_fn, name):
        self._coeff, self.name = coeff_fn, name

    def apply(self, phi, d2, Phi, Phi2, wq):
        if Phi is None:
            return phi, d2, phi.new_zeros(())
        c = self._coeff(phi, Phi, wq)
        return phi - c @ Phi, d2 - c @ Phi2, phi.new_zeros(())


def ProjectionExact():
    return _Projection(project_exact, "projection_exact")


def ProjectionDiagonal():
    return _Projection(project_diagonal, "projection_diagonal(pilot)")
