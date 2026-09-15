"""
experiment_item_capacity.py
Model 1 (Item Memory): recall accuracy as the number of stored items grows,
evaluated with MSE, Cosine Similarity, Bit Accuracy, Exact Match, and Address Accuracy.
Updated with dynamic sub-dataset Pseudoinverse re-fitting and MiniImageNet Dataset (전체 평균 평가).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import config as cfg
from grid_utils import GridCode
from scaffold import GridHPCScaffold
from item_memory import ItemMemory
from datasets import prepare_sensory_data
import matplotlib.pyplot as plt


def extract_matrix(obj):
    if isinstance(obj, np.ndarray): return obj
    if hasattr(obj, 'detach'): return obj.detach().cpu().numpy()
    for attr in ['W', 'weight', 'weights', 'matrix']:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, np.ndarray): return val
            if hasattr(val, 'detach'): return val.detach().cpu().numpy()
    raise TypeError(f"행렬 추출 실패. 타입: {type(obj)}")


def mask_sensory(vec, mask_ratio):
    """원본 fig3b 방식: 이미지(1D로 펼친 sensory pattern)의 마지막
    mask_ratio 비율만큼을 0으로 지운다 (bit-flip 노이즈가 아니라 마스킹)."""
    corrupted = np.array(vec, dtype=float).copy()
    mask_idx = int(len(corrupted) * (1 - mask_ratio))
    corrupted[mask_idx:] = 0
    return corrupted


def salt_and_pepper_sensory(vec, noise_ratio, seed=0):
    """무작위로 고른 noise_ratio 비율의 픽셀을 절반은 최댓값(salt), 절반은
    최솟값(pepper)으로 뒤집는다. mask_sensory와 달리 위치가 매번 무작위이고
    0이 아니라 값 범위의 양 극단으로 튄다."""
    rng = np.random.default_rng(seed)
    corrupted = np.array(vec, dtype=float).copy()
    n = len(corrupted)
    n_noisy = int(n * noise_ratio)
    idx = rng.choice(n, size=n_noisy, replace=False)
    half = n_noisy // 2
    vmin, vmax = corrupted.min(), corrupted.max()
    corrupted[idx[:half]] = vmax   # salt
    corrupted[idx[half:]] = vmin   # pepper
    return corrupted


def apply_noise(vec, noise_type, noise_ratio, seed=0):
    if noise_type == "masking":
        return mask_sensory(vec, noise_ratio)
    elif noise_type == "salt_and_pepper":
        return salt_and_pepper_sensory(vec, noise_ratio, seed=seed)
    raise ValueError(f"unknown noise_type: {noise_type}")


def reshape_sensory_to_image(vec):
    """실제 sensory(이미지) 패턴은 population state가 아니므로 스무딩 없이
    정사각형으로만 접는다 (Ns=3600=60*60이면 원본 이미지 그대로 복원됨)."""
    vec = np.array(vec).flatten()
    n = len(vec)
    side = int(np.ceil(np.sqrt(n)))
    padded = np.zeros(side * side)
    padded[:n] = vec
    return padded.reshape((side, side))


def _stepper_row(widgets, slider, show_slider=True, offset=0):
    """-/+ 버튼 + 숫자 직접 입력 가능한 텍스트박스. IntText는 브라우저 기본 number
    input이라 스피너 화살표 때문에 숫자가 가운데 정렬 안 되는 문제가 있어서, 스피너가
    없는 일반 Text + 수동 int 파싱으로 대체. show_slider=False면 드래그 슬라이더
    자체는 화면에 안 보이고 버튼/텍스트박스만 표시 (slider는 값 저장용으로 계속 씀).
    offset: slider.value(내부, 0-index)와 화면에 보여주는 숫자 사이의 차이 (예: 1이면
    내부 0을 화면엔 "1"로 보여줌 -- item/test index를 1번부터 세고 싶을 때 사용)."""
    from IPython.display import HTML, display as _display
    _display(HTML("<style>.widget-text input{text-align:center !important;}</style>"))
    slider.readout = False
    minus = widgets.Button(description="-", layout=widgets.Layout(width="32px"))
    plus = widgets.Button(description="+", layout=widgets.Layout(width="32px"))
    text = widgets.Text(value=str(slider.value + offset), layout=widgets.Layout(width="60px"))
    minus.on_click(lambda b: setattr(slider, "value", round(max(slider.min, slider.value - slider.step), 10)))
    plus.on_click(lambda b: setattr(slider, "value", round(min(slider.max, slider.value + slider.step), 10)))
    slider.observe(lambda change: setattr(text, "value", str(change["new"] + offset)), names="value")

    def _on_text_change(change):
        try:
            v = int(change["new"]) - offset
        except ValueError:
            return
        slider.value = max(slider.min, min(slider.max, v))

    text.observe(_on_text_change, names="value")
    children = [slider, minus, plus, text] if show_slider else [minus, plus, text]
    return widgets.HBox(children)


def render_node_states_panel(mem, items, target_idx, noise_type, noise_ratio, exp_label=""):
    """show_node_states_interactive의 실제 그림/print 로직 본체(위젯 없이 순수 함수).
    다른 곳(예: 상위 interact)에서 target_idx 등을 직접 넘겨서 재사용할 때 씀."""
    n_iter = 2  # cleanup 고정 (드래그 슬라이더 제거)
    s_original = items[target_idx]
    s_noisy = apply_noise(s_original, noise_type, noise_ratio, seed=target_idx)

    W_hs = extract_matrix(mem.Whs)
    W_sh = extract_matrix(mem.Wsh)
    grid_code = mem.scaffold.grid_code
    g_true = mem.codewords[target_idx]

    h_noisy = W_hs @ s_noisy
    logits_pre_wta = mem.scaffold.Wgh @ h_noisy  # WTA(argmax) 전 module별 실수값 logits
    if n_iter == 0:
        # cleanup 0번 = raw recall(WTA 정화 전 그대로)
        h_clean, g_clean = h_noisy, None
    else:
        h_clean, g_clean = mem.scaffold.cleanup(h_noisy, n_iter=n_iter)  # 실제 recall과 동일한 WTA cleanup
    s_recovered = W_sh @ h_clean

    s_orig_2d = reshape_sensory_to_image(s_original)
    s_noisy_2d = reshape_sensory_to_image(s_noisy)
    s_rec_2d = reshape_sensory_to_image(s_recovered)

    cos_sim = float(np.dot(s_recovered, s_original) /
                     (np.linalg.norm(s_recovered) * np.linalg.norm(s_original) + 1e-10))

    # sensory 코사인 유사도 대신 grid state로 몇 번 item인지 판단: 복원된
    # grid state(g_clean)가 mem.codewords 중 어느 item의 grid state와 정확히
    # 일치하는지 찾음. 어떤 item과도 안 맞으면(=학습 때 배정 안 된 빈 grid state) None.
    if g_clean is None:
        closest_idx = None
    else:
        codewords_arr = np.asarray(mem.codewords)
        match_indices = np.where(np.all(codewords_arr == g_clean, axis=1))[0]
        closest_idx = int(match_indices[0]) if len(match_indices) > 0 else None

    from experiments.experiment_spatial_navigation import plot_grid_modules_square

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))

    axes[0, 0].imshow(s_orig_2d, cmap="gray")
    axes[0, 0].set_title(f"Stored item #{target_idx + 1}", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(s_noisy_2d, cmap="gray")
    axes[0, 1].set_title("Noisy item", fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(s_rec_2d, cmap="gray")
    title3_suffix = "None" if closest_idx is None else f"#{closest_idx + 1}"
    axes[0, 2].set_title(f"Recalled item {title3_suffix} (cos_sim={cos_sim:.2f})", fontsize=12)
    axes[0, 2].axis('off')

    plot_grid_modules_square(axes[1, 0], grid_code, g_true)
    axes[1, 0].set_title("Grid state (true)", fontsize=12)

    plot_grid_modules_square(axes[1, 1], grid_code, logits_pre_wta, vmin=None, vmax=None)
    axes[1, 1].set_title("Grid state (pre-WTA)", fontsize=12)

    if g_clean is not None:
        plot_grid_modules_square(axes[1, 2], grid_code, g_clean)
        axes[1, 2].set_title("Grid state (post-WTA)", fontsize=12)
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# 각 실험과 동일한 파라미터로 mem을 준비하는 함수들
# -------------------------------------------------------------------------
def get_mem_for_Nh_sweep(sbook_flattened, Nh=800, n_items=600):
    scaf_cfg = cfg.DEFAULT_SCAFFOLD
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, Nh=Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    master_items = sbook_flattened.T
    Ns = master_items.shape[1]
    items = master_items[:n_items]
    mem = ItemMemory(grid_code, scaffold, Ns)
    mem.learn(items)
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])
    return mem, items
