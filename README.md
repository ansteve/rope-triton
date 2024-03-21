# rope-triton

RoPE(Rotary Position Embedding)을 Triton으로 작성한 프로젝트입니다.

## 사용법

먼저 CUDA가 설치되어 있어야 합니다. 그 다음 필요한 라이브러리를 아래 명령어로 설치합니다.

```
pip install -r requirements.txt
```

`src/rope_triton.py`에 있는 `apply_rotary_pos_emb_triton` 함수를 사용하면 됩니다.


## 테스트
torch와 triton이 같은 아웃풋을 내는지 확인하기 위해서 아래 스크립트 실행합니다.

```
PYTHONPATH=. pytest tests/test_rope.py
```


## 성능비교




## 참고
* https://github.com/NVIDIA/TransformerEngine/blob/b8eea8aaa94bb566c3a12384eda064bda8ac4fd7/transformer_engine/pytorch/attention.py#L1170-L1230
* https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/fused_rope/fused_rope.cu
