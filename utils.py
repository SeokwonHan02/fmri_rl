"""
utils.py

fMRI_RL 공통 유틸리티 모듈.

주요 기능:
  robust_torch_load  — EOCD 복구 fallback이 포함된 torch.load 래퍼.
                        초기 학습 코드에서 ZIP EOCD 없이 저장된 모델 파일도
                        문제없이 로드한다.
"""

import io
import struct
import zipfile
from pathlib import Path

import torch


# ─── ZIP EOCD 복구 ────────────────────────────────────────────────────────────

def _repair_zip_eocd(path: str) -> bytes:
    """
    ZIP End-of-Central-Directory(EOCD)가 없는 파일을 복구한다.

    PyTorch의 torch.save()는 ZIP 포맷으로 저장하는데, 프로세스가 비정상 종료되거나
    초기 코드에 버그가 있을 때 EOCD 레코드가 누락된 불완전한 파일이 생성될 수 있다.
    이 함수는 Local File Header(LFH) 시그니처를 스캔해 Central Directory와
    EOCD를 재구성한 뒤 완전한 ZIP 바이트를 반환한다.

    주의: PyTorch 체크포인트는 텐서 데이터를 ZIP_STORED(무압축)로 저장하므로
    이 복구 로직이 올바르게 동작한다. DEFLATE 압축 항목의 경우 CRC가 틀릴 수 있다.

    Args:
        path: 복구할 파일 경로

    Returns:
        완전한 ZIP 데이터 (EOCD 포함)
    """
    with open(path, 'rb') as f:
        data = f.read()

    total  = len(data)
    LFH_SZ = 30   # Local File Header 고정 크기 (bytes)
    DD_SZ  = 16   # Data Descriptor 크기 (magic 4 + crc 4 + comp_sz 4 + uncomp_sz 4)

    # LFH 시그니처(PK\x03\x04) 위치 모두 탐색
    lfh_positions = sorted(
        i for i in range(total - 4)
        if data[i:i + 4] == b'PK\x03\x04'
    )

    if not lfh_positions:
        raise RuntimeError(f"No Local File Header signatures found in {path}")

    # 각 LFH에서 메타데이터 파싱
    entries = []
    for pos in lfh_positions:
        try:
            (_, ver, flags, comp, mt, md, _, _, _, fnl, exl) = \
                struct.unpack_from('<4sHHHHHIIIHH', data, pos)
        except struct.error:
            continue
        fn = data[pos + LFH_SZ: pos + LFH_SZ + fnl].decode('utf-8', errors='replace')
        ds = pos + LFH_SZ + fnl + exl   # 실제 데이터 시작 오프셋
        entries.append((pos, fn, ver, flags, comp, mt, md, fnl, exl, ds))

    # 각 항목의 크기·CRC 결정
    file_entries = []
    for i, (pos, fn, ver, flags, comp, mt, md, fnl, exl, ds) in enumerate(entries):
        nxt = entries[i + 1][0] if i + 1 < len(entries) else total

        # Data Descriptor가 있으면 그 값 사용; 없으면 다음 LFH까지의 거리로 추정
        cand = data[nxt - DD_SZ: nxt]
        if cand[:4] == b'PK\x07\x08':
            _, crc, sz, _ = struct.unpack('<4sIII', cand)
        else:
            import zlib
            sz  = nxt - ds
            crc = zlib.crc32(data[ds: ds + sz]) & 0xFFFFFFFF

        # Data Descriptor 플래그 제거 (CRC/크기를 LFH에 직접 기록하므로)
        clean_flags = flags & ~0x08
        file_entries.append((pos, fn, ver, clean_flags, comp, mt, md,
                             crc, fnl, exl, ds, sz))

    # LFH 헤더 내 플래그·CRC·크기 패치
    out = bytearray(data)
    for (pos, fn, ver, flags, comp, mt, md, crc, fnl, exl, ds, sz) in file_entries:
        struct.pack_into('<H', out, pos + 6,  flags)
        struct.pack_into('<I', out, pos + 14, crc)
        struct.pack_into('<I', out, pos + 18, sz)   # compressed size
        struct.pack_into('<I', out, pos + 22, sz)   # uncompressed size (STORED)

    # Central Directory 구성
    cd_off = total
    cd = bytearray()
    for (pos, fn, ver, flags, comp, mt, md, crc, fnl, exl, ds, sz) in file_entries:
        fb = fn.encode('utf-8')
        cd += struct.pack(
            '<4sHHHHHHIIIHHHHHII',
            b'PK\x01\x02', 20, ver, flags, comp, mt, md,
            crc, sz, sz, len(fb), 0, 0, 0, 0, 0, pos,
        ) + fb

    # EOCD 레코드 추가
    n    = len(file_entries)
    eocd = struct.pack(
        '<4sHHHHIIH',
        b'PK\x05\x06', 0, 0, n, n, len(cd), cd_off, 0,
    )

    return bytes(out) + bytes(cd) + bytes(eocd)


# ─── Robust torch.load ────────────────────────────────────────────────────────

def robust_torch_load(path, map_location='cpu', weights_only=False):
    """
    torch.load의 래퍼. EOCD 없이 저장된 파일(구형 모델)도 자동으로 복구·로드한다.

    동작 순서:
      1. 표준 torch.load 시도 → 성공하면 반환.
      2. 실패(BadZipFile, RuntimeError 등)하면 EOCD 복구 후 재시도.
      3. 복구도 실패하면 RuntimeError 발생.

    Args:
        path:          체크포인트 파일 경로 (str 또는 Path)
        map_location:  torch.load의 map_location 인자
        weights_only:  torch.load의 weights_only 인자

    Returns:
        로드된 객체 (보통 state_dict or dict)
    """
    path = str(path)

    # 1차 시도: 표준 로드 (EOCD가 있는 정상 파일)
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception as e:
        first_err = e  # Python 3: 'as' variable is deleted after except block

    # 2차 시도: EOCD 복구 후 로드 (구형 파일 대응)
    print(f"  [robust_torch_load] 표준 로드 실패, EOCD 복구 시도: {Path(path).name}")
    try:
        repaired = _repair_zip_eocd(path)
        buf = io.BytesIO(repaired)
        result = torch.load(buf, map_location=map_location, weights_only=weights_only)
        print(f"  [robust_torch_load] EOCD 복구 성공")
        return result
    except Exception as repair_err:
        raise RuntimeError(
            f"Failed to load checkpoint '{path}'.\n"
            f"  Standard load error : {first_err}\n"
            f"  Repair load error   : {repair_err}"
        ) from repair_err
