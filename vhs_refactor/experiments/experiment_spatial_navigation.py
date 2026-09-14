"""
experiment_spatial_navigation.py
Model 2 (Spatial Memory): fig4c-style demo -- trains a scaffold on a real
random-walk path through image sensory data, builds a novel trajectory that
revisits a few anchors, and evaluates dark path-integration recall/prediction
accuracy at revisited vs unvisited points.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from grid_utils import GridCode
from scaffold import GridHPCScaffold
from spatial_memory import SpatialMemory
from datasets import prepare_sensory_data
from experiments.experiment_item_capacity import reshape_sensory_to_image
import config as cfg

scaf_cfg = cfg.DEFAULT_SCAFFOLD

# ---------------------------------------------------------
# [추가] fig4c 스타일 데모: 실제 이미지 sensory + 실제 경로 시각화 +
# 재방문 복원 / 미방문 trajectory 예측 / sensory -> location 역추론
# ---------------------------------------------------------
# _MAX_NODE_VISITS = 2  # 노드 하나가 이 횟수 넘게 방문되면(8자 한 번은 2회) 경로 통째로 재생성
_MAX_NODE_VISITS = 1  # 8자 크로싱 완전 차단 테스트: 재방문 자체를 금지

_MOVES_DX = np.array([1, -1, 0, 0], dtype=np.int64)
_MOVES_DY = np.array([0, 0, 1, -1], dtype=np.int64)


def _next_seed(seed, attempt):
    """seed가 None이면(고정 안 함) 매 attempt OS 엔트로피로 새 정수 시드를 뽑는다
    -- numba jit 함수는 np.random.seed(int)로만 시드를 받을 수 있어서 필요."""
    if seed is not None:
        return (seed + attempt) % (2**31 - 1)
    return int(np.random.SeedSequence().generate_state(1)[0] % (2**31 - 1))


@njit(cache=True)
def _random_walk_numba(Npos, lo, hi, start_cell, n_steps, max_visits, forbidden_mask, visit_count, seed):
    """self-avoiding + drift-bias random walk 한 번(한 attempt)을 컴파일된 코드로
    실행. cell id = x*Npos+y. status: 0=n_steps 다 채움, 1=(forbidden에 막혀)
    더 갈 곳 없어서 중간에 멈춤(무해, 짧은 경로로 인정), 2=한 칸을 max_visits
    넘게 밟아서 이 attempt 전체가 무효(호출부가 통째로 재시도)."""
    np.random.seed(seed)
    path_cells = np.empty(n_steps + 1, dtype=np.int64)
    path_cells[0] = start_cell
    cur = start_cell
    for step in range(n_steps):
        x = cur // Npos
        y = cur % Npos
        valid_dx = np.empty(4, dtype=np.int64)
        valid_dy = np.empty(4, dtype=np.int64)
        valid_cell = np.empty(4, dtype=np.int64)
        n_valid = 0
        for d in range(4):
            dx = _MOVES_DX[d]
            dy = _MOVES_DY[d]
            nx = x + dx
            ny = y + dy
            if nx < lo or nx >= hi or ny < lo or ny >= hi:
                continue
            ncell = nx * Npos + ny
            if forbidden_mask[ncell]:
                continue
            valid_dx[n_valid] = dx
            valid_dy[n_valid] = dy
            valid_cell[n_valid] = ncell
            n_valid += 1
        if n_valid == 0:
            return path_cells, step, 1

        uv_dx = np.empty(4, dtype=np.int64)
        uv_dy = np.empty(4, dtype=np.int64)
        uv_cell = np.empty(4, dtype=np.int64)
        n_unvisited = 0
        for i in range(n_valid):
            if visit_count[valid_cell[i]] == 0:
                uv_dx[n_unvisited] = valid_dx[i]
                uv_dy[n_unvisited] = valid_dy[i]
                uv_cell[n_unvisited] = valid_cell[i]
                n_unvisited += 1

        if n_unvisited > 0:
            cand_dx, cand_dy, cand_cell, n_cand = uv_dx, uv_dy, uv_cell, n_unvisited
        else:
            cand_dx, cand_dy, cand_cell, n_cand = valid_dx, valid_dy, valid_cell, n_valid

        w = np.empty(n_cand, dtype=np.float64)
        wsum = 0.0
        for i in range(n_cand):
            wi = 1.0 + max(0, cand_dx[i]) * 0.3 + max(0, cand_dy[i]) * 0.3
            w[i] = wi
            wsum += wi
        r = np.random.random() * wsum
        acc = 0.0
        choice = n_cand - 1
        for i in range(n_cand):
            acc += w[i]
            if r <= acc:
                choice = i
                break

        nxt = cand_cell[choice]
        visit_count[nxt] += 1
        cur = nxt
        path_cells[step + 1] = nxt
        if visit_count[nxt] > max_visits:
            return path_cells, step + 1, 2

    return path_cells, n_steps, 0


def build_fig4c_demo(trained_length=100, room_pad=1, seed=None, max_attempts=1000):
    """실제 이미지(prepare_sensory_data)를 실제 random-walk 경로("원래 경로",
    trained path) 위 각 위치에 결합해서 학습한다. VectorHASH_fig4e_random.ipynb의
    fig4c처럼 경로를 Npos x Npos 방 안(벽에서 room_pad칸 이상 떨어진 곳)에
    가둬서, 나중에 만들 novel trajectory가 이 경로를 피해 다닐 공간을
    확보한다. module_periods의 곱(Npos)이 sbook의 Npos(=60)와 같아야
    위치<->이미지 인덱스가 CRT로 정합된다."""
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, Nh=scaf_cfg.Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])

    Npos = int(np.prod(scaf_cfg.module_periods))
    sbook_flattened = prepare_sensory_data()

    lo, hi = room_pad, Npos - room_pad
    forbidden_mask = np.zeros(Npos * Npos, dtype=np.bool_)  # 방 안 walk엔 forbidden 칸 없음, 벽만 제한
    start_cell = (Npos // 2) * Npos + (Npos // 2)

    # 자기 자신을 최대한 안 밟는(self-avoiding) 걸음을 우선 고르고(_random_walk_numba
    # 안 unvisited 타이어), 안 밟은 칸이 하나도 없으면 이미 방문한 칸이라도 밟는다
    # (8자 크로싱 완전 차단 테스트 중이라 엣지 재사용 회피 타이어는 뺐음 -- 위
    # _MAX_NODE_VISITS 주석 참고). fig4e_random.ipynb의 fig4c처럼 drift bias도 적용.
    for attempt in range(max_attempts):
        seed_val = _next_seed(seed, attempt)
        visit_count = np.zeros(Npos * Npos, dtype=np.int64)
        visit_count[start_cell] = 1
        path_cells, n_filled, status = _random_walk_numba(
            Npos, lo, hi, start_cell, trained_length, _MAX_NODE_VISITS, forbidden_mask, visit_count, seed_val
        )
        if status != 2:
            break
    else:
        raise RuntimeError("같은 칸을 너무 자주 밟지 않는 경로를 못 찾았습니다. "
                            "room_pad를 늘리거나 trained_length를 줄여보세요.")
    path_cells = path_cells[: n_filled + 1]  # status==1이면 벽/forbidden에 막혀 조기 종료된 것

    xs = path_cells // Npos
    ys = path_cells % Npos
    path_xy = np.stack([xs, ys], axis=1)  # (n_filled+1, 2), 방 안에 갇힌 물리 좌표
    velocities = [(int(xs[i + 1] - xs[i]), int(ys[i + 1] - ys[i])) for i in range(n_filled)]

    start_indices = [(int(path_xy[0][0]) % k, int(path_xy[0][1]) % k) for k in grid_code.module_periods]
    landmarks = [sbook_flattened[:, int(px) * Npos + int(py)] for px, py in path_xy]

    spatial = SpatialMemory(grid_code, scaffold, sbook_flattened.shape[0])
    spatial.learn(start_indices, velocities, landmarks)
    return dict(grid_code=grid_code, scaffold=scaffold, spatial=spatial,
                sbook_flattened=sbook_flattened, Npos=Npos, room_pad=room_pad,
                start_indices=start_indices, velocities=velocities,
                landmarks=landmarks, path_xy=path_xy)


def _bfs_shortest_path(start, target, forbidden, lo, hi, rng):
    """start->target 4방향 최단 경로. forbidden 칸은 target이 아닌 한 회피.
    rng로 이웃 방문 순서를 섞어서, 실패 시 재시도(다른 seed)하면 다른
    우회로를 찾을 수 있게 한다."""
    from collections import deque
    moves4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == target:
            break
        order = rng.permutation(len(moves4))
        for k in order:
            dx, dy = moves4[k]
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (lo <= nxt[0] < hi and lo <= nxt[1] < hi):
                continue
            if nxt in forbidden and nxt != target:
                continue
            if nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if target not in prev:
        return None
    path = [target]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def build_novel_trajectory(model, novel_length=600, n_overlap=5, seed=None, max_attempts=3000):
    """원래 경로(model)와 정확히 n_overlap개 지점에서만 겹치는("재방문") 새
    경로를 만든다. 원래 경로 위 지정한 n_overlap개(anchor) 위치만 지나가게
    허용하고, 나머지 원래 경로 칸은 전부 회피(forbidden)한다 -- fig4e_random
    .ipynb의 fig4c '겹치는 지점 정확히 N개' 경로 생성 방식과 동일한 원리
    (거기서는 A*, 여기서는 BFS로 forbidden 칸을 피해서 anchor들을 순서대로
    연결). 나머지 길이는 forbidden을 피해서 무작위 보행으로 채운다."""
    Npos, pad = model["Npos"], model["room_pad"]
    trained_path = [tuple(int(v) for v in p) for p in model["path_xy"]]
    lo, hi = pad, Npos - pad

    n_overlap = min(n_overlap, len(trained_path))

    # 자기 자신과 교차하는 random walk라서 서로 다른 step이 같은 물리 좌표를
    # 가리킬 수 있다 -- anchor는 반드시 "서로 다른 좌표"만 뽑아야 실제로
    # forbidden에서 exempt되는 좌표 수가 n_overlap과 일치한다.
    unique_cells = list(dict.fromkeys(trained_path))  # 등장 순서 보존한 중복 제거
    n_overlap = min(n_overlap, len(unique_cells))

    for attempt in range(max_attempts):
        rng = np.random.default_rng(None if seed is None else seed + attempt)
        # anchor 조합 자체가 (조밀한 자기교차 때문에) BFS로 못 뚫는 경우가 있으므로,
        # 매 시도마다 앵커도 다시 뽑는다 (시작점=index 0은 항상 고정).
        if attempt == 0:
            cell_idxs = np.linspace(0, len(unique_cells) - 1, n_overlap, dtype=int)
        else:
            rest = rng.choice(np.arange(1, len(unique_cells)), size=n_overlap - 1, replace=False)
            cell_idxs = np.concatenate([[0], np.sort(rest)])
        anchors = [unique_cells[i] for i in cell_idxs]
        forbidden = set(trained_path) - set(anchors)

        novel_path = [anchors[0]]
        overlap_steps = [0]
        ok = True
        for i in range(len(anchors) - 1):
            seg = _bfs_shortest_path(novel_path[-1], anchors[i + 1], forbidden, lo, hi, rng)
            if seg is None:
                ok = False
                break
            novel_path.extend(seg[1:])
            overlap_steps.append(len(novel_path) - 1)
        if not ok:
            continue

        # anchor 연결은 끝났으니 이제부터는 anchor도 다시 밟으면 안 된다(그러면
        # overlap 수가 n_overlap을 넘어가서 아래 최종 검증에서 매번 재시도로
        # 낭비된다) -- forbidden에 anchor까지 추가해서 패딩 보행이 절대
        # trained path를 다시 안 건드리게 한다.
        forbidden |= set(anchors)

        # 남은 길이는 forbidden을 피해서, build_fig4c_demo와 같은 방식(self-avoiding
        # + drift bias)의 무작위 보행(_random_walk_numba)으로 패딩.
        novel_visit_count = {}
        for cell in novel_path:
            novel_visit_count[cell] = novel_visit_count.get(cell, 0) + 1
        over_limit = any(c > _MAX_NODE_VISITS for c in novel_visit_count.values())

        remaining = novel_length - len(novel_path)
        status = 0 if remaining == 0 else -1
        
        if not over_limit and remaining > 0:
            forbidden_mask = np.zeros(Npos * Npos, dtype=np.bool_)
            for fx, fy in forbidden:
                forbidden_mask[fx * Npos + fy] = True
            visit_count = np.zeros(Npos * Npos, dtype=np.int64)
            for (vx, vy), c in novel_visit_count.items():
                visit_count[vx * Npos + vy] = c

            last_x, last_y = novel_path[-1]
            start_cell = last_x * Npos + last_y
            seed_val = _next_seed(seed, attempt)
            path_cells, n_filled, status = _random_walk_numba(
                Npos, lo, hi, start_cell, remaining, _MAX_NODE_VISITS, forbidden_mask, visit_count, seed_val
            )
            xs = path_cells[1:n_filled + 1] // Npos
            ys = path_cells[1:n_filled + 1] % Npos
            novel_path.extend(zip(xs.tolist(), ys.tolist()))
            if status == 2:
                over_limit = True

        # 최종 검증: 겹치는 지점 수가 정확히 n_overlap이고, 같은 칸을 너무 자주
        # 밟지 않았는지 확인한다. 둘 중 하나라도 어긋나면(예: 패딩 무작위 보행이
        # 우연히 다른 anchor를 또 밟았거나, 한 칸을 3번 넘게 밟았거나) 통째로 재시도한다.
        #if not over_limit and len(set(trained_path) & set(novel_path)) == n_overlap:
        #    break

        # 1. numba 결과에서 status가 0(정상 완주)인지 확인
        is_completed = (status == 0)
        
        # 2. 검증 조건에 추가
        if not over_limit and is_completed and len(novel_path) == novel_length and len(set(trained_path) & set(novel_path)) == n_overlap:
            break
    else:
        raise RuntimeError("겹치지 않는 경로를 못 찾았습니다. n_overlap을 줄이거나 "
                            "trained_length/room_pad를 바꿔보세요.")

    novel_xy = np.array(novel_path)
    novel_velocities = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(novel_path[:-1], novel_path[1:])]
    novel_grid_path = model["grid_code"].path_integrate(model["start_indices"], novel_velocities)

    return dict(novel_xy=novel_xy, novel_velocities=novel_velocities,
                novel_grid_path=novel_grid_path, overlap_steps=overlap_steps)


def plot_paths(model, novel_model=None, title="Grid world", unvisited_steps=None,
               show_revisit_labels=True, current_step=None):
    """원래 경로(검정)와 새 경로(파랑)를 x-y 평면에 같이 그리고, 재방문
    지점(겹치는 위치)을 빨간 원으로 표시한다. unvisited_steps를 주면
    (미방문 step 인덱스들) 초록 X로 추가 표시.
    show_revisit_labels=False면 재방문 지점의 "t=n" 라벨을 끈다.
    current_step을 주면 trained path 위 그 step 위치를 검은 다이아몬드로 표시."""
    from matplotlib.ticker import MultipleLocator
    path_xy = model["path_xy"]
    path_linewidth = 1.5
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.plot(path_xy[0, 0], path_xy[0, 1], "o", color="black", markersize=4, label="start location")
    ax.plot(path_xy[:, 0], path_xy[:, 1], "-", color="dimgray", alpha=1.0, linewidth=path_linewidth,
            label="original path")
    if current_step is not None:
        cx, cy = path_xy[current_step]
        ax.plot(cx, cy, "D", color="black", markersize=5, label=f"current (t={current_step})")

    if novel_model is not None:
        novel_xy = novel_model["novel_xy"]
        ax.plot(novel_xy[:, 0], novel_xy[:, 1], "-", color="blue", linewidth=path_linewidth, label="new path")
        overlap_steps = novel_model["overlap_steps"]
        overlap = novel_xy[overlap_steps]
        # t=0은 항상 novel path의 시작점(=trained path 시작점)이라 "start location"과
        # 겹침 -- revisit 표시는 t=0 빼고, 라벨(t=0)은 유지.
        non_start = [t != 0 for t in overlap_steps]
        if any(non_start):
            ax.plot(overlap[non_start, 0], overlap[non_start, 1], "o", color="red", markersize=4,
                    label="revisit location")
        if show_revisit_labels:
            for t, (px, py) in zip(overlap_steps, overlap):
                ax.annotate(f"t={t}", xy=(px, py), xytext=(3, 3), textcoords="offset points",
                            fontsize=8, color="firebrick")
        if unvisited_steps:
            unvisited = novel_xy[list(unvisited_steps)]
            ax.plot(unvisited[:, 0], unvisited[:, 1], "x", color="orange", markersize=6, mew=2,
                    label="novel location")
            if show_revisit_labels:
                for t, (px, py) in zip(unvisited_steps, unvisited):
                    ax.annotate(f"t={t}", xy=(px, py), xytext=(3, 3), textcoords="offset points",
                                fontsize=8, color="darkorange")

    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def plot_grid_modules_square(ax, grid_code, g, cmap="OrRd", vmin=0, vmax=1, shear_deg=60):
    """한 axes(정사각형) 안에 모듈별 grid state를 가로로 나란히 그린다. 박스 크기를
    period(k)에 비례시켜서 모듈마다 칸(cell) 하나의 화면 크기는 동일하게 유지하고,
    n×n 격자 자체의 전체 크기만 k가 클수록 커지게 한다. shear_deg만큼 x축 방향으로
    기울여서(pcolormesh 좌표를 직접 shear) 마름모(실제 grid cell lattice) 느낌을 낸다."""
    ax.axis("off")
    blocks = grid_code.state_blocks(g)
    periods = grid_code.module_periods
    n = len(blocks)
    theta = np.deg2rad(shear_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 겹침을 고정 폭(예: -0.03)이 아니라 "모듈 자기 폭의 일정 비율"로 잡아야, 작은
    # 모듈(4x4)과 큰 모듈(7x7)의 겹침 정도가 시각적으로 동일해 보인다 -- 고정폭이면
    # 큰 모듈일수록 같은 겹침이 상대적으로 작아 보여서 간격이 들쭉날쭉해 보였다.
    margin = 0.02
    overlap_ratio = 0.3
    K_all = sum(periods)
    K_prefix = sum(periods[:-1])  # 마지막 모듈 뒤에는 겹칠 다음 모듈이 없음
    # cell 한 변 길이 s를 기준으로 두 basis vector (s,0), (s*cos_t, s*sin_t) 둘 다
    # 길이가 s로 같다 -- 즉 변 길이가 전부 같은 진짜 마름모(rhombus) 타일링.
    s = (1 - 2 * margin) / ((1 + cos_t) * (K_all - overlap_ratio * K_prefix))
    x0 = margin
    min_x, max_x, max_h = x0, x0, 0
    for idx, (k, block) in enumerate(zip(periods, blocks)):
        # block은 [x phase, y phase] 순서(encode_state 참고). 행=y, 열=x 좌표로 맞추려고
        # block.T를 쓰고, (i,j) 격자점을 i*(s,0) + j*(s*cos_t, s*sin_t)로 배치해서
        # 변 길이가 전부 s로 같은 마름모 격자를 만든다 (실제 grid cell 논문 도식 방식).
        jj, ii = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        X = x0 + ii * s + jj * s * cos_t
        Y = jj * s * sin_t
        ax.pcolormesh(X, Y, block.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                      edgecolors="dimgray", linewidth=0.3)
        w_x, w_y = k * s * (1 + cos_t), k * s * sin_t
        corners = [(x0, 0), (x0 + k * s, 0), (x0 + k * s + k * s * cos_t, w_y), (x0 + k * s * cos_t, w_y), (x0, 0)]
        ax.plot(*zip(*corners), color="dimgray", linewidth=0.8)
        ax.text(x0 + w_x / 2, w_y + 0.03, f"{k}×{k}", fontsize=7, color="dimgray",
                ha="center", va="bottom")
        max_h = max(max_h, w_y)
        min_x = min(min_x, x0, x0 + k * s * cos_t)
        max_x = max(max_x, x0 + k * s, x0 + k * s + k * s * cos_t)
        x0 += w_x * (1 - overlap_ratio) if idx < n - 1 else w_x

    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(0, max_h + 0.04)
    ax.set_aspect("equal")


def _unvisited_steps(model, novel_model):
    """novel path의 step 인덱스 중 실제 좌표가 trained path 좌표(전부, anchor 포함)와
    한 번도 안 겹치는 step만 골라낸다. overlap_steps만으로 거르면 부족하다 --
    anchor를 이어붙인 뒤 남는 길이를 채우는 무작위 보행(padding) 구간이 같은
    anchor 좌표를 다시 밟아도 overlap_steps에는 안 잡히기 때문.
    novel path 자체도 self-crossing(같은 좌표를 여러 step에서 반복 방문)이라,
    좌표 기준으로도 중복 제거해서 좌표별로 처음 등장한 step 하나만 남긴다 --
    안 그러면 near/far/무작위 선택 결과가 서로 같은 좌표를 여러 번 뽑아서
    지도/패널에서 겹쳐 보인다."""
    trained_cells = set(tuple(int(v) for v in p) for p in model["path_xy"])
    novel_xy = novel_model["novel_xy"]
    seen = set()
    steps = []
    for t in range(len(novel_xy)):
        cell = tuple(int(v) for v in novel_xy[t])
        if cell in trained_cells or cell in seen:
            continue
        seen.add(cell)
        steps.append(t)
    return steps


def _plot_novel_step_column(axes, col, t, model, novel_model):
    """novel path의 step t 지점에서 True/Recall sensory, grid state, HPC state를
    한 컬럼(axes[:, col])에 그린다. demo_revisit_predictions/demo_unvisited_predictions
    공용 -- 재방문/미방문 어느 지점이든 t만 주면 동일하게 동작."""
    spatial, scaffold = model["spatial"], model["scaffold"]
    sbook_flattened, Npos = model["sbook_flattened"], model["Npos"]

    g = novel_model["novel_grid_path"][t]
    h = scaffold.grid_to_hpc(g)
    h_clean, g_clean = scaffold.cleanup(h, n_iter=2)
    recon = spatial.Wsh.recall(h_clean)

    x, y = novel_model["novel_xy"][t]
    target = sbook_flattened[:, (int(x) % Npos) * Npos + (int(y) % Npos)]

    cos = float(np.dot(target, recon) / (np.linalg.norm(target) * np.linalg.norm(recon) + 1e-10))

    axes[0, col].imshow(reshape_sensory_to_image(target), cmap="gray")
    axes[0, col].set_title(f"t={t}", fontsize=11, fontweight="bold"); axes[0, col].axis("off")
    axes[1, col].imshow(reshape_sensory_to_image(recon), cmap="gray")
    axes[1, col].set_title(f"cos={cos:.3f}", fontsize=10); axes[1, col].axis("off")

    plot_grid_modules_square(axes[2, col], model["grid_code"], g_clean)
    axes[3, col].imshow(reshape_sensory_to_image(h_clean), cmap="magma")
    axes[3, col].set_title("HPC state", fontsize=9, color="dimgray")
    axes[3, col].axis("off")
    return cos


def _plot_novel_steps_grid(model, novel_model, selected, suptitle, show_row_labels=True):
    n_rows = 4  # True sensory / Recall sensory / grid state(모듈 전체, 한 패널) / HPC state
    n_show = len(selected)
    # grid state(row 2) 실제 내용은 sheared 마름모라 가로세로 비율이 ~3:1 (넓적함).
    # 다른 행(정사각형 이미지)과 같은 높이를 주면 aspect="equal" 유지 시 위아래에
    # 안 쓰이는 흰 여백이 크게 남는다 -- 그 행만 내용 비율(1/3)에 맞게 낮춰서
    # 모양(마름모) 그대로 유지하면서 여백만 없앤다.
    row_h_ratio = [1, 1, 1 / 3, 1]
    fig, axes = plt.subplots(n_rows, n_show, gridspec_kw={"height_ratios": row_h_ratio},
                              figsize=(3.2 * n_show, 3.2 * sum(row_h_ratio)))
    if n_show == 1:
        axes = axes.reshape(n_rows, 1)

    cos_list = [_plot_novel_step_column(axes, col, t, model, novel_model)
                for col, t in enumerate(selected)]

    if show_row_labels:
        row_labels = ["True", "Recall", "Grid state", "HPC state"]
        for row, label in enumerate(row_labels):
            axes[row, 0].annotate(label, xy=(-0.15, 0.5), xycoords="axes fraction",
                                   ha="right", va="center", fontsize=12, fontweight="bold",
                                   rotation=90)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    # tight_layout()은 axes마다 get_tightbbox(텍스트 렌더링)를 다 계산해야 해서
    # 열 수가 많아지면(4x4=16 axes) 느림(~0.5s/call). 고정 여백으로 대체해서
    # 슬라이더 인터랙션마다 다시 그릴 때 빠르게.
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.02, hspace=0.08, wspace=0.25)
    plt.show()
    return cos_list


def demo_revisit_predictions(model, novel_model, n_revisits=10, seed=2):
    """새 경로가 원래 경로와 겹치는(재방문) 지점들 중 최대 n_revisits개를
    뽑아서, 그 지점에서 예측되는 sensory를 실제 landmark와 비교한다
    (fig4c cell 6 스타일: 위 True / 아래 Recall). 이 지점들은 학습 때 실제로
    결합(binding)됐던 위치라 recall이 잘 됨."""
    overlap_steps = [t for t in novel_model["overlap_steps"] if t != 0]  # t=0은 시작점(=novel path 시작)과
    if not overlap_steps:                                                # 겹치는 자명한 지점이라 제외
        return []

    rng = np.random.default_rng(seed)
    n_show = min(n_revisits, len(overlap_steps))
    selected = np.sort(rng.choice(overlap_steps, size=n_show, replace=False))
    suptitle = "Recalled images on revisited locations"
    return _plot_novel_steps_grid(model, novel_model, selected, suptitle)


def plot_unvisited_distance_map(model, novel_model, n_show=2, seed=2):
    """미방문 지점들 중 n_show개를 무작위로 골라 지도에 표시 (재방문 지점은
    빨간 원). demo_unvisited_by_distance에서 쓸 unvisited step 목록을 반환."""
    candidates = _unvisited_steps(model, novel_model)
    if not candidates:
        return []

    rng = np.random.default_rng(seed)
    n_show = min(n_show, len(candidates))
    unvisited = sorted(rng.choice(candidates, size=n_show, replace=False).tolist())

    plot_paths(model, novel_model, title="Grid world", unvisited_steps=unvisited)
    return unvisited


def demo_unvisited_by_distance(model, novel_model, unvisited):
    """plot_unvisited_distance_map이 뽑아준 미방문 지점들의 sensory recall을 보여준다.
    학습 때 결합(binding)이 없던 위치라 재방문 지점(cos_sim~1.0)보다 recall이
    부정확함을 확인할 수 있다."""
    if not unvisited:
        return []
    return _plot_novel_steps_grid(model, novel_model, unvisited, "Recalled images on novel locations",
                                   show_row_labels=False)


def demo_unvisited_predictions(model, novel_model, n_show=10, seed=2):
    """새 경로가 원래 경로와 한 번도 안 겹치는(학습 때 결합이 전혀 없었던) 지점들
    중 최대 n_show개를 뽑아서 같은 방식으로 확인한다. grid state는 dark
    path-integration만으로도 여전히 정확하지만(모듈 shift가 결정론적이라서),
    그 위치의 sensory는 Whs/Wsh가 한 번도 학습한 적이 없으므로 recall이 실제
    landmark와 얼마나 다른지(대체로 낮은 cos_sim) 보여준다."""
    candidates = _unvisited_steps(model, novel_model)
    if not candidates:
        print("미방문 지점이 없습니다. novel_length를 늘려보세요.")
        return []

    rng = np.random.default_rng(seed)
    n_show = min(n_show, len(candidates))
    selected = np.sort(rng.choice(candidates, size=n_show, replace=False))
    suptitle = f"Novel path UNVISITED points ({n_show}/{len(candidates)} points shown)"
    return _plot_novel_steps_grid(model, novel_model, selected, suptitle)


def demo_sensory_to_location(model, query_step=5):
    """sensory 입력 -> 어느 위치(grid code)인지 역추론 (Whs로 s->h, scaffold cleanup으로 h->g)."""
    spatial, grid_code, scaffold = model["spatial"], model["grid_code"], model["scaffold"]
    s_query = model["landmarks"][query_step]
    h_cue = spatial.Whs.recall(s_query)
    _, g_clean = scaffold.cleanup(h_cue, n_iter=2)
    predicted_indices = grid_code.decode_state(g_clean)

    true_path = grid_code.path_integrate(model["start_indices"], model["velocities"])
    true_indices = grid_code.decode_state(true_path[query_step])
    true_xy = tuple(model["path_xy"][query_step])
    print(f"[sensory -> location] query_step={query_step} | true_xy={true_xy} | "
          f"predicted module idx={predicted_indices} | true module idx={true_indices} | "
          f"match={predicted_indices == true_indices}")
    return predicted_indices, true_indices



if __name__ == "__main__":
    run()