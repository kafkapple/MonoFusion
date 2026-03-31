# MonoFusion Architecture — SE(3) Motion Bases

> Fact-checked against `flow3d/scene_model.py`, `flow3d/params.py` (2026-03-29)

---

## 1. Why SE(3) Bases, Not MLP?

The common assumption is that MonoFusion uses an MLP deformation field (like D-NeRF or HexPlane). **This is wrong.** MonoFusion uses explicit **SE(3) basis trajectories** — a structured parametric representation that:

- Encodes motion as a linear combination of K rigid-body transforms
- Is time-indexed (not position-indexed) → globally shared across all Gaussians
- Supports analytically exact 3D scene flow extraction without forward passes

### Why Not MLP?

| Approach | Pros | Cons |
|----------|------|------|
| **MLP deformation** (D-NeRF style) | Flexible, smooth | Per-point forward pass; can't decompose into interpretable bases |
| **SE(3) basis trajectories** (MonoFusion) | Interpretable, fast; shared bases = multi-view consistent | Limited by K bases; assumes motion factorizes |

**Multi-view consistency** emerges naturally: all cameras share the same K basis SE(3) trajectories. Even with independently-seeded per-camera RAFT tracks, the optimizer must fit a single set of motion bases that explains all views simultaneously.

---

## 2. Core Data Structures

From `flow3d/params.py`:

```python
class MotionBases:
    params["rots"]   # (K, T, 6)  — K basis rotations at each time, 6D continuous rep
    params["transls"]# (K, T, 3)  — K basis translations at each time

class GaussianParams:
    params["means"]        # (G, 3)  — canonical positions (learned)
    params["motion_coefs"] # (G, K)  — per-Gaussian softmax blending weights
```

**K = number of motion bases** (default: 10)
**T = number of frames** (60 for M5t2 PoC)
**G = number of Gaussians** (5K FG + 10K BG)

---

## 3. Transform Computation

From `flow3d/scene_model.py`:

```python
def compute_transforms(ts: Tensor, coefs: Tensor) -> Tensor:
    """
    Args:
        ts:    (B,) frame indices
        coefs: (G, K) per-Gaussian softmax weights
    Returns:
        transfs: (G, B, 3, 4) — per-Gaussian SE(3) transforms
    """
    # For each basis k:
    #   rot_k(t)   = rot6d_to_matrix(rots[k, t])    → (B, 3, 3)
    #   transl_k(t)= transls[k, t]                   → (B, 3)
    #   SE3_k(t)   = [rot_k | transl_k]              → (B, 3, 4)
    #
    # Weighted combination (per-Gaussian linear blend):
    #   transfs_i = Σ_k coefs[i, k] * SE3_k(t)
```

**Softmax normalization**: `coefs = softmax(motion_coefs, dim=-1)` ensures weights sum to 1 per Gaussian. Each Gaussian "belongs" softly to a combination of motion bases.

**6D rotation representation**: Uses 6D continuous rotation (`rot6d_to_matrix`) — no gimbal lock, differentiable everywhere (unlike Euler angles or quaternions).

---

## 4. Gaussian Position at Time t

```python
def compute_poses_fg(ts: Tensor) -> tuple[Tensor, Tensor]:
    """
    Returns per-Gaussian means and covariances at time t.

    means_i(t) = SE3_combined_i(t) @ [canonical_μ_i; 1]
               = (Σ_k coefs[i,k] · rot_k(t)) @ μ_i + (Σ_k coefs[i,k] · transl_k(t))
    """
    coefs   = self.gaussians.get_coefs()         # (G, K) softmax
    transfs = self.motion_bases.compute_transforms(ts, coefs)  # (G, B, 3, 4)

    canonical = self.gaussians.params["means"]   # (G, 3)
    # Homogeneous: append 1 → (G, 4)
    # Apply transform: (G, B, 3, 4) @ (G, 4) → (G, B, 3) = means at time ts
    means_t = transfs @ F.pad(canonical, (0, 1), value=1)[..., None]  # (G, B, 3)
    return means_t
```

---

## 5. 3D Scene Flow Extraction

**Definition**: Frame-to-frame displacement of each Gaussian's canonical position.

```python
# After training, extract scene flow analytically:
def extract_scene_flow(model, frame_t: int) -> np.ndarray:
    """Returns flow vectors [G, 3] = means(t+1) - means(t)."""
    ts  = torch.tensor([frame_t],     device=model.device)
    ts1 = torch.tensor([frame_t + 1], device=model.device)

    means_t  = model.compute_poses_fg(ts)   # (G, 1, 3)
    means_t1 = model.compute_poses_fg(ts1)  # (G, 1, 3)

    flow = (means_t1 - means_t).squeeze(1)  # (G, 3)
    return flow.cpu().numpy()
```

**Physical meaning**: `flow[i]` is the 3D velocity (displacement/frame) of Gaussian `i` at time `t`.

For a mouse: paws have high magnitude, body has moderate, tail has directional flow.

---

## 6. Choosing K (Number of Motion Bases)

**Default**: K = 10 (MonoFusion paper default, also appropriate for M5t2)

**Intuition**: K controls how many independent rigid-body motion primitives the model can learn.

For a mouse skeleton:
- Head (1 DOF), neck (1), spine (2), pelvis (1), L/R forelimbs (2), L/R hindlimbs (2) → ~9 semantic bases
- K = 10 is a natural fit

**Sensitivity**:
| K | Risk | Use case |
|---|------|----------|
| K < 6 | Under-parameterized: limb independence lost | Rigid objects |
| K = 10 | Good for mice, dogs, quadrupeds | **M5t2 default** |
| K > 20 | Overfitting; slow convergence | Articulated hands, full body |

**Rule of thumb**: K ≈ degrees of freedom of the object's kinematic chain × 1.5

---

## 7. Key Design Implications

### 7.1 Motion Bases Are NOT Per-Camera

The same K bases are shared across all cameras. RAFT tracks from cam00/cam01/cam03 all supervise the **same** `rots(K,T,6)` and `transls(K,T,3)`. This is why:

- Per-camera track quality differences are OK — the optimizer averages across cameras
- cam02 contributing 0 FG visible tracks doesn't corrupt motion bases (other cameras fill in)
- Multi-view consistency is **structurally enforced**, not just regularized

### 7.2 Background Gaussians

Background Gaussians have `motion_coefs` that converge to low-magnitude motion (effectively static). They are NOT excluded from motion learning — they simply learn near-zero flow.

### 7.3 Temporal Resolution

`rots` and `transls` have explicit per-frame parameters (shape `[K, T, ...]`). This means:
- The model must see supervision at every frame (or interpolate via temporal smoothness)
- Frames with 0 visible tracks (e.g., cam02 at all frames) → motion bases are NOT supervised by that camera for those frames
- Mitigation: other cameras + smoothness regularization fill the gap

---

## 8. Relationship to 4D-GS

MonoFusion is **not** the standard "4D Gaussian Splatting" (Wu et al. 2024 = Gaussian + HexPlane + MLP). It uses motion bases which are:

- Closer to **SC-GS** (Sparse Controlled Gaussian Splatting, 2024) in structure
- More interpretable than HexPlane (each basis has physical meaning)
- Faster inference (no per-frame MLP forward pass)

---

↑ MOC: `docs/README.md` | ↔ Related: `scene_flow_and_tracking.md`, `core_architecture.md`, `training_guide.md`

*MonoFusion Architecture — SE(3) Motion Bases | MonoFusion M5t2 PoC | 2026-03-29*
