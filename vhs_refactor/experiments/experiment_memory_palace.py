"""
experiment_memory_palace.py

VectorHASH_fig7.ipynb 에 실제로 정의된 함수들
(build_seq_scaffold, make_hairpin_path, path_to_indices,
 recall_sequence_once, pseudotrain_Wps/Wsp, module_wise_NN_2d, nonlin)
을 그대로 사용해서 Memory Palace(3. 시퀀스 기반 sensory <-> mnemonic 연상 기억) 데모를 구현.

주의:
- build_seq_scaffold 내부는 numpy.random 전역 상태(randn/randint)를 사용하므로
  완전한 재현성이 필요하면 호출 전에 np.random.seed(...)를 직접 설정하세요.
- src.assoc_utils_np / src.assoc_utils_np_2D / src.seq_utils 등은
  노트북과 동일한 경로에 있다고 가정합니다.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint

from src.assoc_utils_np import *
from src.assoc_utils_np_2D import gen_gbook_2d, module_wise_NN_2d
from src.seq_utils import *
from src.sensory_utils import *
from src.senstranspose_utils import *


# =========================================================================
# 노트북에서 그대로 가져온 공통 유틸 (수정 없음)
# =========================================================================
def build_seq_scaffold(lambdas, Nh, gamma=0.6, thresh=2.5, nruns=1):
    Ng = int(np.sum(np.square(lambdas)))
    Npos = int(np.prod(lambdas))
    Nstates = Npos * Npos

    gbook = gen_gbook_2d(lambdas, Ng, Npos)
    gbook_flat = gbook.reshape(Ng, Nstates)

    Wpg = randn(nruns, Nh, Ng)

    prune = int((1 - gamma) * Nh * Ng)
    mask = np.ones((Nh, Ng))
    mask[
        randint(low=0, high=Nh, size=prune),
        randint(low=0, high=Ng, size=prune)
    ] = 0
    Wpg = Wpg * mask

    pbook = nonlin(np.einsum("ijk,klm->ijlm", Wpg, gbook), thresh=thresh)
    pbook_flat = pbook.reshape(nruns, Nh, Nstates)

    Wgp = train_gcpc(pbook_flat, gbook_flat, Nstates)

    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]

    return {
        "lambdas": lambdas, "Ng": Ng, "Nh": Nh, "Npos": Npos, "Nstates": Nstates,
        "gbook": gbook, "gbook_flat": gbook_flat, "pbook": pbook, "pbook_flat": pbook_flat,
        "Wpg": Wpg, "Wgp": Wgp, "module_sizes": module_sizes,
        "module_gbooks": module_gbooks, "gamma": gamma, "thresh": thresh,
    }




def make_hairpin_path(width, height, x0=0, y0=0):
    path = []
    for y in range(height):
        xs = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
        for x in xs:
            path.append((x + x0, y + y0))
    return np.array(path, dtype=int)


def path_to_indices(path_locations, Npos):
    return np.array([x * Npos + y for x, y in path_locations], dtype=int)




def recall_sequence_once(scaf, S_seq, P_seq, Nseq, rng=None, noise_frac=0.0, skip_cleanup=False,
                          S_query=None, return_grid=False):
    """
    노트북의 두 버전(무노이즈 버전 / noise_frac 버전)을 하나로 통합.
    - rng=None 또는 noise_frac=0 : 완전 결정론적(무노이즈) 회상
      -> 항목 7의 "full-rank recalled sensory states" 조건에 해당
    - rng가 주어지고 noise_frac>0 : grid 표상(gin)에 소량 가우시안 노이즈를 섞음
      -> 항목 3의 "반복 회상" 실험에서 두 번의 독립 회상을 만들 때 사용
    - S_query : Wps/Wsp는 원래(깨끗한) S_seq로 학습하되, 실제 query로는 이 배열을
      사용 -> "노이즈 낀 사진을 보여줬을 때도 제대로 회상하는가"를 테스트할 때
      2a의 apply_noise(masking/salt_and_pepper)로 만든 이미지를 넣어주면 됨.
      None이면 S_seq를 그대로 query로 사용(무노이즈).
    - skip_cleanup=True : module_wise_NN_2d(모듈별 discrete cleanup)를 건너뛰고
      noise 낀 연속값 gin을 그대로 사용 -> 노이즈가 실제로 얼마나 표상을
      흐트러뜨리는지 시각화할 때 사용 (cleanup이 이걸 대부분 지워버리기 때문에,
      cleanup 이후 결과만 보면 노이즈 효과가 잘 안 보임).
    """
    Wps = pseudotrain_Wps(P_seq, S_seq, Nseq)
    Wsp = pseudotrain_Wsp(S_seq, P_seq, Nseq)

    S_query = S_seq if S_query is None else S_query

    pin = nonlin(Wps @ S_query, thresh=0)
    gin = scaf["Wgp"] @ pin

    if rng is not None and noise_frac > 0:
        noise_std = noise_frac * gin.std()
        gin = gin + rng.normal(scale=noise_std, size=gin.shape)

    if skip_cleanup:
        G_rec = gin
    else:
        G_rec = np.zeros((1, scaf["Ng"], Nseq))
        for k in range(Nseq):
            G_rec[:, :, k] = module_wise_NN_2d(
                gin[:, :, k, None], scaf["module_gbooks"], scaf["module_sizes"]
            )[0, :, 0]

    P_rec = nonlin(scaf["Wpg"] @ G_rec, scaf["thresh"])
    S_rec = Wsp @ P_rec
    return (S_rec, G_rec) if return_grid else S_rec


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# =========================================================================
# 1~5: palace 길이별 old-landmark 열화 + 반복회상 일관성 + item 결합 A/B 비교
# =========================================================================


# =========================================================================
# 6: Nh, Ns, Cs(=lambdas 조합), item 개수(=palace 길이) 스윕
# =========================================================================


# =========================================================================
# 7: full-rank 회상 조건에서 P_exact ~= min(Cs, Ns) 검증
# =========================================================================

# =========================================================================
# fig 7d: VectorHASH_fig7de.ipynb의 fig7d를 그대로 재현 (image-panel 데모)
# =========================================================================
def make_embedded_image_book_for_fig7(
    Ns, Nstates, Npos,
    block_x0=0, block_y0=0, block_w=60, block_h=60,
    seed=0,
    shuffle_images=False,
    use_tanh_inverse=True,
):
    """VectorHASH_fig7.py의 make_embedded_image_book_for_fig7과 동일.
    실제 MiniImageNet 이미지를 block_w x block_h 위치 블록에 심어 넣고,
    나머지 위치는 무작위 sensory 패턴으로 채운다."""
    rng = np.random.default_rng(seed)

    npy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "BW_miniimagenet_4600_60_60.npy")
    img = np.load(npy_path)
    img_flat = img.reshape(img.shape[0], 3600).T
    # ponytail: float32로 낮춰서 메모리 절반 (Render 512MB 한도, 데모용이라 정밀도 손실 무해)
    # cast를 먼저 해야 float64 원본(img)이 살아있는 동안 float64 중간 복사본까지
    # 추가로 안 생김 (- 후 cast하면 float64 중간값이 잠깐 더 떠서 메모리 튐)
    img_flat = img_flat.astype(np.float32)
    del img
    img_flat -= img_flat.mean()

    if shuffle_images:
        perm = rng.permutation(img_flat.shape[1])
        img_flat = img_flat[:, perm]

    if use_tanh_inverse:
        smin, smax = np.amin(img_flat), np.amax(img_flat)
        scale = 1.9 / (smax - smin)
        shift = -0.95 - smin * scale
        img_flat *= scale
        img_flat += shift  # np.interp((smin,smax)->(-0.95,0.95))와 동일한 선형 변환, in-place
        np.arctanh(img_flat, out=img_flat)
        img_embed = img_flat
    else:
        img_embed = np.sign(img_flat)
        smin, smax = None, None

    n_positions = block_w * block_h
    # block이 (0,0)부터 시작해서 Nstates 전체를 정확히 덮는 경우(이 앱의 실제 사용
    # 패턴) idx(=x*Npos+y)가 항상 k와 같은 순서로 증가 -> sbook_full 앞부분은 img_embed의
    # 앞 n_positions열과 순서대로 같다. (예전엔 랜덤 배열 만들고 한 칸씩 덮어썼는데
    # 전부 버려지는 값이라 낭비였음 + 계산도 전체 3600열에 대해 다 했음)
    # 실제 이미지는 3600장뿐 -- n_positions이 이를 넘으면 초과분은 이미지 재사용(중복)
    # 대신 원본 fig7 스크립트처럼 무작위 sensory 패턴으로 채운다(원본 이미지 유일성 유지).
    assert block_x0 == 0 and block_y0 == 0 and n_positions == Nstates and Npos == block_h
    n_real = min(n_positions, img_embed.shape[1])
    sbook_full = rng.standard_normal((img_embed.shape[0], n_positions)).astype(np.float32)
    sbook_full[:, :n_real] = img_embed[:, :n_real]

    return sbook_full, smin, smax






def plot_palace_path(block_w=13, block_h=4, block_x0=0, block_y0=0, show_labels=True, highlight_t=None):
    """3a의 plot_paths처럼, hairpin 경로(=palace에 이미지를 저장하는 순서)를
    x-y 평면에 그린다. 각 점이 t번째로 결합된 위치(=idxs_7d[t])와 대응.
    highlight_t를 주면 그 위치를 큰 별표로 강조 표시(현재 슬라이더 t 위치 등)."""
    path = make_hairpin_path(block_w, block_h, block_x0, block_y0)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(path[:, 0], path[:, 1], "-o", markersize=1.2, alpha=0.6, color="tab:blue")
    ax.plot(path[0, 0], path[0, 1], "gs", markersize=4, label="start (t=0)")
    ax.plot(path[-1, 0], path[-1, 1], "r^", markersize=4, label=f"end (t={len(path) - 1})")
    if highlight_t is not None:
        hx, hy = path[highlight_t]
        ax.plot(hx, hy, "*", color="orange", markersize=8, markeredgecolor="black",
                label=f"current (t={highlight_t})", zorder=5)
    if show_labels:
        for t, (x, y) in enumerate(path):
            ax.annotate(str(t), xy=(x, y), xytext=(2, 2), textcoords="offset points", fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Palace hairpin trajectory ({block_w}x{block_h} = {len(path)} items)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()
    return path
