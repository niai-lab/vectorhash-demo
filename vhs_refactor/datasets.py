"""
datasets.py
Synthetic data generators shared across experiments.
"""
import numpy as np
import os

_sensory_data_cache = None


def prepare_sensory_data():
    """사용자가 제공한 MiniImageNet 전처리 로직 기반 데이터 로더"""
    # 입력이 고정 파일 하나뿐이라(랜덤성 없음) 결과가 항상 동일 -> 모듈 레벨로 캐싱.
    # report.ipynb에서 데모 여러 개(2a/3a/4a/4b)가 각자 이 함수를 불러서 52MB짜리
    # 배열을 매번 새로 만들면 Render 512MB 한도에서 누적으로 OOM남.
    global _sensory_data_cache
    if _sensory_data_cache is not None:
        return _sensory_data_cache

    # print("\n=== MiniImageNet 데이터 로드 및 전처리 시작 ===")

    # 파라미터 세팅
    Ns = 3600
    Npos = 60  
    n_states = Npos * Npos  # 3600
    
    block_x0 = 0
    block_y0 = 0
    block_w = 60
    block_h = 60
    
    # 1. 이미지 로드 (cwd에 상관없이 이 폴더(vhs_refactor) 안의 파일만 사용)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'BW_miniimagenet_4600_60_60.npy')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

    img = np.load(file_path)
    # print("원본 이미지 형태:", img.shape)

    # 2. 평탄화 및 전치 (n_images, 60, 60) -> (3600, n_images)
    img_flat = img.reshape((img.shape[0], 3600)).T

    # 3. Sensory book 구성: block(60x60)이 n_states(3600) 전체를 정확히 덮어서
    # idx(=x*Npos+y)가 항상 k와 같은 순서로 증가 -> 결과가 img_flat과 완전히 동일.
    # (예전엔 random 배열을 만들고 한 칸씩 덮어썼는데 전부 버려지는 값이라 낭비였음)
    # ponytail: float32로 낮춰서 메모리 절반 (512MB Render free tier 한도 때문에 필요, 정밀도 손실은 데모용이라 무해)
    assert block_x0 == 0 and block_y0 == 0 and block_w == Npos and block_h == Npos
    sbook_flattened = img_flat.astype(np.float32)
    del img, img_flat

    # 4. 연상 기억을 위한 전처리 (스케일링 및 비선형 변환) - 전부 in-place로 진행해서
    # 중간 복사본이 동시에 여러 개 메모리에 떠있지 않게 함
    bw_mean = np.mean(sbook_flattened)
    sbook_flattened -= bw_mean

    sbookmin = np.amin(sbook_flattened)
    sbookmax = np.amax(sbook_flattened)

    scale = 1.9 / (sbookmax - sbookmin)
    shift = -0.95 - sbookmin * scale
    sbook_flattened *= scale
    sbook_flattened += shift  # np.interp((sbookmin,sbookmax)->(-0.95,0.95))와 동일한 선형 변환

    np.arctanh(sbook_flattened, out=sbook_flattened)

    _sensory_data_cache = sbook_flattened
    return sbook_flattened