import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")  # 제한 없으면 OpenBLAS가 nproc(72)까지 스레드 스폰 -> 코어 수 넘어가면서 오버헤드 폭발 (48/72스레드 실측 6~7배 느려짐). 16~32 구간은 실측상 서로 비슷해서 24로 고정 (numpy import 전에 설정해야 함)
os.environ.setdefault("OMP_NUM_THREADS", "24")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 200  # 모든 figure에 공통 적용, compact하게

import streamlit as st

import config as cfg

# report.ipynb의 1/2번 데모(Item Memory, Spatial Memory)가 공유하던 scaffold 설정.
# experiment_spatial_navigation.py는 이 값을 모듈 최초 import 시점에 한 번만
# 스냅샷(scaf_cfg = cfg.DEFAULT_SCAFFOLD)하므로, 그 import보다 먼저 설정해야 함.
cfg.DEFAULT_SCAFFOLD = cfg.ScaffoldConfig(
    module_periods=[3, 4, 5], Nh=400,
    connection_prob=0.6, threshold=0.5, nonlinearity="relu_threshold",
)

from PIL import Image
from experiments.experiment_item_capacity import (
    prepare_sensory_data, get_mem_for_Nh_sweep, render_node_states_panel, apply_noise,
)
from experiments.experiment_spatial_navigation import (
    build_fig4c_demo, build_novel_trajectory,
    demo_revisit_predictions, demo_unvisited_by_distance, plot_unvisited_distance_map,
    plot_grid_modules_square,
)
from experiments.experiment_memory_palace import (
    build_seq_scaffold, make_embedded_image_book_for_fig7,
    make_hairpin_path, path_to_indices, recall_sequence_once, cos_sim,
)
from src.assoc_utils_np import pseudotrain_Wps, pseudotrain_Wsp
from grid_utils import GridCode

st.set_page_config(page_title="Vector-HaSH demo", layout="wide")

# 슬라이더(stepper_slider)들 사이 상하 간격 압축: 기본 vertical block gap이 넓어서
# 슬라이더 여러 개 쌓이면 세로로 길어짐.
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] { gap: 0.4rem; }
div[data-testid="stHorizontalBlock"] { gap: 0.4rem; }
button p { font-size: 0.75rem; }
button[data-testid^="stBaseButton"] { padding-left: 0.25rem; padding-right: 0.25rem; min-width: 0; }
.st-key-item_memory_fig div[data-testid="stImage"] { max-width: 65% !important; margin-left: auto !important; margin-right: auto !important; }
.st-key-item_memory_fig div[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
.st-key-spatial_memory_figs div[data-testid="stImage"] { max-width: 90% !important; margin-left: auto !important; margin-right: auto !important; }
.st-key-spatial_memory_figs div[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
</style>
""", unsafe_allow_html=True)


def _render_open_figures():
    """plt.show()를 st.pyplot()으로 리다이렉트: 기존 실험 코드(plot 함수들)를
    수정 없이 그대로 재사용하기 위한 monkeypatch.
    use_container_width=False -- 기본값(True)은 컨테이너 폭에 무조건 맞춰
    늘려버려서 figsize를 줄여도 화면 출력 크기가 그대로였음. False로 두면
    figsize*dpi가 실제 출력 픽셀 크기가 되고(컨테이너보다 크면 그 폭까지만
    줄어듦), figsize가 다시 출력 크기를 제어하는 값이 된다.
    plt.get_fignums()로 열린 figure 전부를 돌면, matplotlib의 전역 figure
    registry가 프로세스 전체(세션/스레드 공유)라서 다른 세션이 아직 안 닫은
    stray figure까지 같이 렌더링해버리는 경우가 있었다(예: Spatial Memory의
    Grid world 자리에 Item Memory 그림이 나오는 현상). 각 plot 함수는 항상
    plt.show() 호출 전에 figure를 정확히 하나만 만들므로, 방금 만든
    plt.gcf() 하나만 그리면 다른 세션/직전 호출의 잔여 figure와 안 섞인다.
    bbox_inches=None -- st.pyplot 기본값(bbox_inches="tight")은 저장되는 PNG
    크기를 figsize가 아니라 실제 렌더된 콘텐츠(row label 유무, suptitle 텍스트
    길이 등)를 감싸는 bbox로 정하기 때문에, figsize/내용 개수(n_show)가 같아도
    패널마다 콘텐츠가 다르면 최종 픽셀 크기가 미세하게 달라져서 화면에 나란히
    놓았을 때 세로 길이가 달라 보였다. None으로 두면 figsize 그대로 저장되어
    같은 figsize는 항상 같은 픽셀 크기가 된다."""
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=False, bbox_inches=None)
    plt.close(fig)


plt.show = _render_open_figures


def _stepper_bump(key, delta, min_value, max_value):
    # on_click 콜백은 위젯 인스턴스화 전(rerun 직전)에 실행되므로 여기서 session_state를
    # 고쳐야 StreamlitWidgetAlreadyInstantiatedError 없이 안전하게 값이 바뀐다.
    st.session_state[key] = min(max(st.session_state[key] + delta, min_value), max_value)


def stepper_slider(label, min_value, max_value, value, step, key, container=None):
    """st.slider + 양옆 -/+ 버튼. key로 session_state에 값 보관.
    container를 넘기면(예: st.columns(2)의 한 칸) 그 안에 렌더링 -- 한 줄에 슬라이더
    여러 개를 나란히 배치할 때 사용."""
    container = container if container is not None else st
    if key not in st.session_state:
        st.session_state[key] = value
    st.session_state[key] = min(max(st.session_state[key], min_value), max_value)

    c1, c2, c3 = container.columns([10, 1, 1], gap="small")
    with c2:
        st.button("➖", key=f"{key}_minus", use_container_width=True,
                  on_click=_stepper_bump, args=(key, -step, min_value, max_value))
    with c3:
        st.button("➕", key=f"{key}_plus", use_container_width=True,
                  on_click=_stepper_bump, args=(key, step, min_value, max_value))
    with c1:
        st.slider(label, min_value, max_value, step=step, key=key)
    return st.session_state[key]


# =========================================================================
# 1. Item Memory (구 2a)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Training item memory...")
def _get_item_mem(Nh, n_items_sub):
    sbook_flattened = prepare_sensory_data()
    return get_mem_for_Nh_sweep(sbook_flattened, Nh=Nh, n_items=n_items_sub)


def render_item_memory():
    st.header("1. Item Memory")
    col1, col2 = st.columns(2)
    Nh = stepper_slider("$N_h$", 200, 800, cfg.DEFAULT_SCAFFOLD.Nh, 10, key="item_Nh", container=col1)
    n_items_sub = stepper_slider("$N_s$", 1, 1000, cfg.DEFAULT_SCAFFOLD.Nh // 2 + 1, 10, key="item_Ns", container=col2)
    col3, col4 = st.columns(2)
    idx_label = stepper_slider("Item index", 1, n_items_sub, 1, 1, key="item_idx", container=col3)
    noise_ratio = stepper_slider("Noise ratio", 0.0, 0.9, 0.1, 0.1, key="item_noise_ratio", container=col4)
    noise_type = st.selectbox("Noise type", ["masking", "salt_and_pepper"], index=1)

    mem_sub, items_sub = _get_item_mem(Nh, n_items_sub)
    target_idx = min(idx_label - 1, n_items_sub - 1)
    with st.container(key="item_memory_fig"):
        render_node_states_panel(mem_sub, items_sub, target_idx, noise_type, noise_ratio)


# =========================================================================
# 2. Spatial Memory (구 3a)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Building trained path model...")
def _get_3a_model(trained_length):
    return build_fig4c_demo(trained_length=trained_length)


_SPATIAL_N_OVERLAP = 3  # 시작점 포함 재방문 지점 수 고정


@st.cache_resource(max_entries=1, show_spinner="Building novel trajectory...")
def _get_3a_novel(_model, trained_length, novel_length):
    return build_novel_trajectory(_model, novel_length=novel_length, n_overlap=_SPATIAL_N_OVERLAP)


def render_spatial_memory():
    st.header("2. Spatial Memory")
    col1, col2 = st.columns(2)
    trained_length = stepper_slider("Original path length:", 20, 100, 50, 10, key="spatial_trained_length", container=col1)
    novel_length = stepper_slider("New path length:", 20, 100, 50, 10, key="spatial_novel_length", container=col2)

    model = _get_3a_model(trained_length)
    novel_model = _get_3a_novel(model, trained_length, novel_length)

    with st.container(key="spatial_memory_figs"):
        map_col, revisit_col, novel_col = st.columns(3)
        with map_col:
            unvisited = plot_unvisited_distance_map(model, novel_model, n_show=2)
        with revisit_col:
            demo_revisit_predictions(model, novel_model, n_revisits=_SPATIAL_N_OVERLAP)
        with novel_col:
            demo_unvisited_by_distance(model, novel_model, unvisited)


# =========================================================================
# Memory Palace가 사용하는 데이터 소스: miniimagenet(old item) / 숫자카드(new item).
# lambdas=(2,5,7), Ns=3600, seed=0으로 둘 다 동일해서 한 번만 계산해 공유한다.
# Npos=70 -> Nstates=4900 (N_m 슬라이더 최대 4000을 커버하기 위해 (2,3,5)에서 확장).
# =========================================================================
_PALACE_LAMBDAS = (2, 5, 7)
_PALACE_NS = 3600
_PALACE_SEED = 0


def _load_number_card_images_grayscale(numbers):
    """number_card_60x60/ 폴더(1~999, 미리 렌더링해둔 PNG)에서 numbers에 해당하는
    카드만 그레이스케일로 불러와 (Ns, len(numbers)) 형태로 반환."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "number_card_60x60")
    imgs = [np.array(Image.open(os.path.join(folder, f"{n:03d}.png")).convert("L"), dtype=np.float64)
            for n in numbers]
    arr = np.stack(imgs, axis=0)
    return arr.reshape(len(numbers), -1).T


@st.cache_data(max_entries=1, show_spinner="Rendering number cards...")
def _make_numbered_card_book(Ns, Nstates, Npos, block_w, block_h, seed):
    """block_w*block_h개 위치마다 1..n_positions 숫자 카드를 중복 없이 랜덤 순서로
    배치한다. 카드가 전부 유일하므로 워터마크 없이도 pinv가 항상 full rank."""
    rng = np.random.default_rng(seed)
    img_h = img_w = int(round(np.sqrt(Ns)))
    assert img_h * img_w == Ns

    n_positions = block_w * block_h
    numbers = rng.permutation(n_positions) + 1
    img_flat = _load_number_card_images_grayscale(numbers).astype(np.float32)
    img_flat += rng.standard_normal(img_flat.shape).astype(np.float32) * 10.0
    img_flat -= img_flat.mean()

    smin, smax = np.amin(img_flat), np.amax(img_flat)
    scale = 1.9 / (smax - smin)
    shift = -0.95 - smin * scale
    img_flat *= scale
    img_flat += shift
    np.arctanh(img_flat, out=img_flat)

    assert n_positions == Nstates and Npos == block_h
    return img_flat


@st.cache_data(max_entries=1, show_spinner="Loading miniimagenet book...")
def _get_palace_books():
    Npos = int(np.prod(_PALACE_LAMBDAS))
    block_w = block_h = Npos
    Nstates = Npos * Npos
    sbook_old, _, _ = make_embedded_image_book_for_fig7(
        _PALACE_NS, Nstates, Npos, 0, 0, block_w, block_h,
        seed=_PALACE_SEED, shuffle_images=False, use_tanh_inverse=True,
    )
    mbook_new = _make_numbered_card_book(_PALACE_NS, Nstates, Npos, block_w, block_h, _PALACE_SEED)
    path_all = make_hairpin_path(block_w, block_h, 0, 0)
    idxs_all = path_to_indices(path_all, Npos)
    return sbook_old, mbook_new, idxs_all, block_w, block_h


@st.cache_resource(max_entries=1, show_spinner="Training scaffold...")
def _get_palace_scaffold(Nh, gamma=0.6, thresh=0.5):
    return build_seq_scaffold(list(_PALACE_LAMBDAS), Nh, gamma=gamma, thresh=thresh, nruns=1)


# =========================================================================
# 3. Memory Palace: Cleanup Test (구 4b)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Running cleanup pipeline...")
def _get_4b_pipeline(Nh, depth):
    scaf = _get_palace_scaffold(Nh)
    sbook_old, mbook_new, idxs_all, _, _ = _get_palace_books()
    idxs_seq = idxs_all[:depth]
    P_seq = scaf["pbook_flat"][:, :, idxs_seq]
    S_seq = sbook_old[:, idxs_seq]
    M_seq = mbook_new[:, idxs_seq]

    # Wps/Wsp는 S_seq/P_seq/depth에만 의존(어느 item을 조회하든 동일)하므로 여기서
    # 한 번만 학습해 반환 -- _recover()가 item index 바뀔 때마다 이 pinv(depth 전체
    # 크기)를 다시 돌리지 않고 재사용하기 위함.
    Wps = pseudotrain_Wps(P_seq, S_seq, depth)
    Wsp = pseudotrain_Wsp(S_seq, P_seq, depth)
    S_clean, G_clean = recall_sequence_once(scaf, S_seq, P_seq, depth, np.random.default_rng(1),
                                             return_grid=True, Wps=Wps, Wsp=Wsp)
    S_addr = np.sign(S_clean[0])
    Wms = M_seq @ np.linalg.pinv(S_addr)          # 주소 -> new item
    Wsm_raw = S_seq @ np.linalg.pinv(M_seq)       # new item -> sensory (원본 스케일)
    G_true = scaf["gbook_flat"][:, idxs_seq]
    return scaf, S_seq, M_seq, P_seq, S_clean, Wms, Wsm_raw, G_clean, G_true, Wps, Wsp


@st.cache_data(show_spinner=False)
def _recover(_scaf, _P_seq, _M_seq, _Wms, _Wsm_raw, _Wps, _Wsp, Nh, depth, t, noise_ratio_vis):
    # item index(t)만 바뀔 때마다 recall_sequence_once가 depth 전체를 재학습(pinv)하고
    # depth개 위치 전부 cleanup 루프를 도는 게 느려서 (Nh, depth, t, noise_ratio_vis)
    # 기준으로 캐싱 + t 하나짜리 컬럼만 회상하도록 축소.
    # Wps/Wsp(_get_4b_pipeline에서 이미 학습됨)는 S_query가 뭐든 동일한 선형사상이라
    # t 하나만 조회할 때도 재학습 없이 그대로 재사용 가능(depth 무관, O(1)).
    # 배열 인자(_ 접두사)는 해시 대상에서 제외되므로, 캐시 키 구분을 위해
    # Nh/depth를 별도 인자로 받는다(안 그러면 Nh만 바뀌어도 이전 파이프라인의
    # 결과가 잘못 재사용될 수 있음).
    true_item = _M_seq[:, t]
    noisy_item = true_item if noise_ratio_vis == 0.0 else apply_noise(true_item, "salt_and_pepper", noise_ratio_vis, seed=2)
    sensory_est_noisy = _Wsm_raw @ noisy_item
    S_query_col = sensory_est_noisy[:, None]
    S_rec, G_rec = recall_sequence_once(_scaf, None, _P_seq, 1, np.random.default_rng(1),
                                         S_query=S_query_col, return_grid=True, Wps=_Wps, Wsp=_Wsp)
    sensory_cleaned = S_rec[0, :, 0]
    addr_clean = np.sign(sensory_cleaned)
    item_rec = _Wms @ addr_clean
    return noisy_item, sensory_est_noisy, sensory_cleaned, item_rec, G_rec[0, :, 0]


def render_memory_palace_b():
    st.header("3. Memory Palace")
    Ns = _PALACE_NS
    img_h = img_w = int(round(np.sqrt(Ns)))
    _, _, idxs_all, _, _ = _get_palace_books()
    n_cards_full = len(idxs_all)

    col1, col2 = st.columns(2)
    Nh = stepper_slider("$N_h$", 10, 400, 200, 5, key="palace_b_Nh", container=col1)
    n_cards = stepper_slider("$N_m$", 101, min(4000, n_cards_full), min(1000, n_cards_full), 50,
                              key="palace_b_Nm", container=col2)
    col3, col4 = st.columns(2)
    depth = stepper_slider("$N_s$", 2, n_cards, min(30, n_cards), 1, key="palace_b_depth", container=col3)
    noise_ratio_vis = stepper_slider("Noise ratio", 0.0, 0.9, 0.3, 0.1, key="palace_b_noise_ratio", container=col4)
    t = stepper_slider("Item index", 1, depth, 1, 1, key="palace_b_idx") - 1

    scaf, S_seq, M_seq, P_seq, S_clean, Wms, Wsm_raw, G_clean, G_true, Wps, Wsp = _get_4b_pipeline(Nh, depth)

    true_sensory = S_seq[:, t]
    sensory_baseline_rec = S_clean[0, :, t]
    true_item = M_seq[:, t]
    noisy_item, sensory_est_noisy, sensory_cleaned, item_rec, g_cleanup = _recover(
        scaf, P_seq, M_seq, Wms, Wsm_raw, Wps, Wsp, Nh, depth, t, noise_ratio_vis)

    panels = [
        (true_sensory, f"Stored item #{t + 1}", G_true[:, t]),
        (sensory_baseline_rec, f"Recalled item #{t + 1} (cos_sim={cos_sim(sensory_baseline_rec, true_sensory):.2f})", G_clean[0, :, t]),
        (true_item, "Mnemonic item", None),
        (noisy_item, "Noisy mnemonic item", None),
        (sensory_est_noisy, f"Noisy item recon (cos_sim={cos_sim(sensory_est_noisy, true_sensory):.2f})", None),
        (sensory_cleaned, f"Cleanup item recall #{t + 1} (cos_sim={cos_sim(sensory_cleaned, true_sensory):.2f})", g_cleanup),
        (item_rec, f"Recalled mnemonic item (cos_sim={cos_sim(item_rec, true_item):.2f})", None),
    ]
    grid_code = GridCode(module_periods=list(_PALACE_LAMBDAS))
    fig, axes = plt.subplots(2, len(panels), figsize=(3.1 * len(panels), 6.8))
    for col, (vec, title, g) in enumerate(panels):
        axes[0, col].imshow(vec.reshape(img_h, img_w), cmap="gray")
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        if g is None:
            axes[1, col].axis("off")
        else:
            plot_grid_modules_square(axes[1, col], grid_code, g)
            axes[1, col].set_title("Grid state", fontsize=9)
    plt.tight_layout()
    plt.show()


# =========================================================================
st.title("Vector-HaSH")

section = st.sidebar.radio(
    "Section",
    ["1. Item Memory", "2. Spatial Memory", "3. Memory Palace"],
)

if section == "1. Item Memory":
    render_item_memory()
elif section == "2. Spatial Memory":
    render_spatial_memory()
else:
    render_memory_palace_b()
