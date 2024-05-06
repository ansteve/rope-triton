# rope-triton

RoPE(Rotary Position Embedding)을 Triton으로 작성한 프로젝트입니다.


## 성능비교

Torch 버젼과 JAX 버젼의 성능 비교를 하기 위해서 아래 스크립트 실행합니다.

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 rope_jax/benchmark_jax.py
```

jax.jit을 사용하지 않았을때는 속도는 아래와 같았습니다.

```
   seq_length     Torch    Triton       Jax
0       256.0  0.134784  0.044000  2.683440
1       512.0  0.211232  0.078592  2.664448
2      1024.0  0.392192  0.147488  2.660704
3      2048.0  0.735136  0.288896  2.471200
```

jax.jit을 사용했을때의 속도는 아래와 같았습니다.

```
   seq_length     Torch    Triton       Jax
0       256.0  0.134944  0.050944  0.132064
1       512.0  0.212048  0.077568  0.143296
2      1024.0  0.392512  0.147520  0.187856
3      2048.0  0.734464  0.287424  0.278624
```
