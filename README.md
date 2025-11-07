# Multi-Head Attention in Keras

Keras/TensorFlow 구현된 Multi-Head Attention 레이어입니다. "Attention is All You Need" 논문(Vaswani et al., 2017)을 기반으로 구현되었습니다.

## 주요 기능

- **MultiHeadAttention**: 완전한 multi-head attention 메커니즘
- **TransformerBlock**: attention과 feed-forward network를 포함한 완전한 transformer 블록
- **Masking 지원**: padding mask와 look-ahead mask 지원
- **유연한 사용**: self-attention, cross-attention 모두 가능

## 파일 구조

```
.
├── multi_head_attention.py  # Multi-head attention 구현
├── example_usage.py          # 사용 예제
└── README.md                 # 문서
```

## Multi-Head Attention이란?

Multi-head attention은 Transformer 아키텍처의 핵심 구성 요소입니다. 모델이 서로 다른 위치의 다른 표현 부분공간(representation subspaces)의 정보에 동시에 주목할 수 있게 합니다.

### 작동 원리

1. **Linear Projections**: Query(Q), Key(K), Value(V)를 각각 선형 변환
2. **Split into Heads**: d_model 차원을 num_heads개의 작은 차원으로 분할
3. **Scaled Dot-Product Attention**: 각 헤드에서 독립적으로 attention 계산
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k)V
   ```
4. **Concatenate**: 모든 헤드의 출력을 연결
5. **Final Linear**: 최종 선형 변환

### 수식

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

## 설치 및 요구사항

```bash
pip install tensorflow>=2.0.0
pip install numpy
```

## 사용 방법

### 기본 사용

```python
from multi_head_attention import MultiHeadAttention
import tensorflow as tf

# 파라미터 설정
d_model = 512      # 모델 차원
num_heads = 8      # attention head 수
seq_len = 10       # 시퀀스 길이
batch_size = 2

# 입력 생성
query = tf.random.normal((batch_size, seq_len, d_model))
key = tf.random.normal((batch_size, seq_len, d_model))
value = tf.random.normal((batch_size, seq_len, d_model))

# Multi-head attention 레이어 생성
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Forward pass
output, attention_weights = mha(query, key, value)

print(f"Output shape: {output.shape}")  # (2, 10, 512)
```

### Self-Attention

```python
# Self-attention: Q, K, V가 모두 동일
x = tf.random.normal((batch_size, seq_len, d_model))
output, attn_weights = mha(x, x, x)
```

### Cross-Attention (Encoder-Decoder Attention)

```python
# Cross-attention: Query는 decoder에서, Key/Value는 encoder에서
encoder_output = tf.random.normal((batch_size, encoder_len, d_model))
decoder_input = tf.random.normal((batch_size, decoder_len, d_model))

output, attn_weights = mha(
    query=decoder_input,
    key=encoder_output,
    value=encoder_output
)
```

### Masking 사용

```python
from multi_head_attention import create_look_ahead_mask

# Look-ahead mask (decoder self-attention용)
mask = create_look_ahead_mask(seq_len)
mask = mask[tf.newaxis, tf.newaxis, :, :]

output, attn_weights = mha(x, x, x, mask=mask)
```

### Transformer Block 사용

```python
from multi_head_attention import TransformerBlock

# Transformer block 생성
transformer = TransformerBlock(
    d_model=512,
    num_heads=8,
    dff=2048,      # Feed-forward network 차원
    dropout=0.1
)

# Forward pass
output = transformer(x, training=True)
```

## 예제 실행

```bash
python example_usage.py
```

다음 예제들이 포함되어 있습니다:
1. 기본 multi-head attention 사용
2. Self-attention
3. Masking을 사용한 attention
4. 완전한 Transformer block
5. 시퀀스 분류 모델 구축
6. Cross-attention

## API 문서

### MultiHeadAttention

```python
MultiHeadAttention(d_model, num_heads, dropout=0.1)
```

**Parameters:**
- `d_model` (int): 모델의 차원 (embedding dimension)
- `num_heads` (int): attention head의 수 (d_model은 num_heads로 나누어떨어져야 함)
- `dropout` (float): dropout 비율 (기본값: 0.1)

**Input:**
- `query`: shape (batch_size, seq_len_q, d_model)
- `key`: shape (batch_size, seq_len_k, d_model)
- `value`: shape (batch_size, seq_len_v, d_model)
- `mask`: (선택) mask tensor

**Output:**
- `output`: shape (batch_size, seq_len_q, d_model)
- `attention_weights`: shape (batch_size, num_heads, seq_len_q, seq_len_k)

### TransformerBlock

```python
TransformerBlock(d_model, num_heads, dff, dropout=0.1)
```

**Parameters:**
- `d_model` (int): 모델의 차원
- `num_heads` (int): attention head의 수
- `dff` (int): feed-forward network의 내부 차원
- `dropout` (float): dropout 비율

**Input:**
- `x`: shape (batch_size, seq_len, d_model)
- `mask`: (선택) mask tensor
- `training`: (선택) training 모드 여부

**Output:**
- shape (batch_size, seq_len, d_model)

## 이론적 배경

### Attention 메커니즘

Attention 메커니즘은 입력 시퀀스의 각 위치에서 다른 모든 위치의 정보를 가중합하여 출력을 생성합니다.

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

- **Q (Query)**: "무엇을 찾고 있는가"
- **K (Key)**: "무엇이 있는가"
- **V (Value)**: "실제 정보"
- **√d_k로 스케일링**: gradient 안정화

### Multi-Head의 장점

1. **다양한 관점**: 각 헤드가 다른 표현 부분공간을 학습
2. **병렬 처리**: 모든 헤드를 동시에 계산
3. **더 풍부한 표현**: 여러 위치의 정보를 동시에 활용

## 하이퍼파라미터 가이드

### 일반적인 설정

| 모델 크기 | d_model | num_heads | dff |
|----------|---------|-----------|-----|
| Base | 512 | 8 | 2048 |
| Small | 256 | 4 | 1024 |
| Large | 1024 | 16 | 4096 |

### 선택 가이드

- **d_model**: 작업의 복잡도에 따라 결정 (512가 일반적)
- **num_heads**: 보통 8 또는 16 (d_model의 약수여야 함)
- **dff**: 일반적으로 4 * d_model
- **dropout**: 0.1이 일반적, overfitting 시 증가

## 성능 최적화 팁

1. **배치 크기**: GPU 메모리가 허용하는 한 크게
2. **시퀀스 길이**: 필요한 만큼만 사용 (O(n²) 복잡도)
3. **Mixed Precision**: TensorFlow의 mixed precision 사용
4. **Gradient Checkpointing**: 메모리가 부족할 때 사용

## 참고 문헌

- Vaswani et al., "Attention is All You Need", NeurIPS 2017
- [Original Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

## 라이선스

MIT License

## 기여

이슈나 Pull Request는 언제든 환영합니다!
