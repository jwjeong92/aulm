# Proposal: Hypothesis-Driven Sensitive Block Classification and Adaptive ECC Cost Distribution

## 1. Problem Definition

우리가 풀고 싶은 문제는 "어떤 비트가 중요한가" 자체가 아니다.

진짜 문제는 다음과 같다.

1. 양자화된 초대형 모델의 가중치 저장 구조가 주어져 있다.
2. 실제 보호 단위는 `ECC codeword` 혹은 `quantization group`이다.
3. 각 보호 단위마다 bit-flip에 대한 민감도를 정량화하고 싶다.
4. ECC mode마다 `parity cost`와 `residual BER`가 다르다.
5. 전체 ECC 예산 제약 하에서, 어떤 block에 어떤 ECC mode를 배정해야 전체 위험이 최소가 되는지 결정하고 싶다.

중요한 제약은 다음과 같다.

- 대상 모델은 `175B` 같은 초대형 모델일 수 있다.
- 따라서 반복적인 fault injection, seed sweep, 다량의 calibration 실험에 의존하면 안 된다.
- 방법은 `한 번의 계측 + 한 번의 집계 + 한 번의 최적화`로 설명 가능해야 한다.

즉, 우리가 원하는 것은:

`bit ranking -> 실험 반복`


이 아니라

`analytic exposure model -> block risk -> ECC allocation`


이다.

## 2. Core Hypothesis

핵심 가설은 두 부분으로 나뉜다.

### 2.1 Model-Side Hypothesis

한 weight의 bit flip이 만드는 위험도는 대략 다음 세 가지의 곱으로 설명될 수 있다.

1. 양자화 scale
2. activation sensitivity proxy
3. bit significance

즉, 어떤 bit flip이 위험한 이유는:

- 그 bit가 높은 자리수라서 perturbation이 크고
- 그 weight가 속한 quantization group의 scale이 크고
- 그 weight가 연결된 activation column이 민감하기 때문이다

### 2.2 Hardware-Side Hypothesis

ECC는 "민감도" 자체를 바꾸지 않는다.
ECC는 오직 "오류가 남는 확률", 즉 residual BER만 바꾼다.

따라서:

- model-side sensitivity
- hardware-side protection quality

를 분리해서 모델링할 수 있다.

이 분리가 성립하면, 초대형 모델에서도 반복 실험 없이 보호 예산 배분이 가능해진다.

## 3. Sensitivity Formulation

### 3.1 Per-Weight Bit Exposure

weight $w$가 decoder layer $ell$에 있고, input column이 $c(w)$, quantization group이 $g(w)$에 속한다고 하자.

다음 기호를 둔다.

- $h_{c(w)}$: activation sensitivity proxy
- $s_{g(w)}$: 해당 quantization group의 scale
- $\alpha_{ell}$: optional layer amplification factor
- $k$: bit position

Unsigned integer 양자화에서 clipping 경계 효과를 무시하면:

$$\text
q'_k = q \space \oplus \space 2^k \\
\delta q_k = q'_k - q \\
|\delta q_k| = 2^k \\
|\delta w_k| = s_{g(w)} \cdot 2^k \\
$$

유도는 간단하다.

- bit $k$를 XOR로 뒤집으면 정수 표현에서 정확히 $2^k$만큼 변화한다.
- floating-point weight perturbation은 quantization scale을 곱한 값이므로 $\delta w_k = s_{g(w)} \cdot \delta q_k$다.
- 따라서 magnitude의 제곱은:

$$
(\delta w_k)^2 = s_{g(w)}^2 \cdot (\delta q_k)^2 = s_{g(w)}^2 \cdot 4^k
$$

따라서 per-weight bit exposure를 다음과 같이 둔다.

$$
e_w(k) = (1/2) \cdot \alpha_{ell} \cdot h_{c(w)} \cdot s_{g(w)}^2 \cdot 4^k
$$

이 식은 다음 local second-order proxy에서 바로 나온다.

$$
\delta L_w(k) \approx (1/2) \cdot \alpha_{ell} \cdot h_{c(w)} \cdot (\delta w_k)^2
$$

여기에 $(\delta w_k)^2 = s_{g(w)}^2 \cdot 4^k$를 대입하면:

$$
e_w(k)
\approx (1/2) \cdot \alpha_{ell} \cdot h_{c(w)} \cdot (\delta w_k)^2
= (1/2) \cdot \alpha_{ell} \cdot h_{c(w)} \cdot s_{g(w)}^2 \cdot 4^k
$$

즉 $1/2$는 second-order Taylor coefficient이고, $h_{c(w)}$는 해당 column의 curvature-like proxy이며, $s_{g(w)}^2 \cdot 4^k$는 실제 perturbation magnitude를 담당한다.

이 식은 "bit k가 뒤집혔을 때 그 weight가 만들어낼 local expected damage mass"를 나타낸다.

### 3.2 Quantization Group Exposure

quantization group $g$에 속한 모든 weight에 대해 exposure를 합치면:

$$
E_g(k) = \sum_{w \in g} e_w(k)
$$

이는 group 단위 risk를 constituent weight들의 additive exposure 합으로 보는 정의다. 위 식에 $e_w(k)$를 대입하면:

$$
E_g(k)
= \sum_{w \in g} (1/2) \cdot \alpha_{ell}(w) \cdot h_{c(w)} \cdot s_{g(w)}^2 \cdot 4^k
$$

만약 group 내 scale이 shared 된다면:

$$
E_g(k) = (1/2) \cdot \alpha_{ell}(g) \cdot s_g^2 \cdot 4^k \cdot \sum_{w \in g} h_{c(w)}
$$

이는 group 내부에서 $\alpha_{ell}(w) = \alpha_{ell}(g)$이고 $s_{g(w)} = s_g$이므로 공통항을 밖으로 뺀 결과다.

$$
E_g(k)
= \sum_{w \in g} (1/2) \cdot \alpha_{ell}(g) \cdot h_{c(w)} \cdot s_g^2 \cdot 4^k
= (1/2) \cdot \alpha_{ell}(g) \cdot s_g^2 \cdot 4^k \cdot \sum_{w \in g} h_{c(w)}
$$

이 형태는 매우 직관적이다.

group이 위험한 이유는:

- $s_g^2$가 커서 bit flip perturbation이 크고
- $\sum h$가 커서 activation sensitivity mass가 크고
- $\alpha_{layer}$가 크면 downstream amplification이 더 크기 때문이다

### 3.3 ECC Codeword Exposure

실제 메모리 보호 단위는 codeword이므로, packing map $Pi$를 통해 weight들을 codeword $b$로 매핑한다.

그러면 codeword exposure는:

$$
E_b(k) = \sum_{w \in b} e_w(k)
$$

이 역시 codeword 안에 실제로 packed된 weight들의 exposure를 모두 합친 정의다.

또는 group 관점에서는:

$$
E_b(k) = \sum_{g \space overlaps \space b} overlap(g, b, k)
$$

여기서 $overlap(g, b, k)$를

$$
overlap(g, b, k) = \sum_{w \in g \cap b} e_w(k)
$$

로 두면, codeword 안의 weight 집합을 group별 부분집합으로 분해한 것과 같다. 즉:

$$
E_b(k)
= \sum_{w \in b} e_w(k)
= \sum_{g \space overlaps \space b} \sum_{w \in g \cap b} e_w(k)
= \sum_{g \space overlaps \space b} overlap(g, b, k)
$$

이다.

이 값이 바로 hardware allocation이 실제로 바라봐야 하는 block sensitivity이다.

## 4. ECC Mode and Residual Risk

ECC mode $m$에 대해 bit별 residual BER 벡터를 다음과 같이 둔다.

$$
p_k^{m}, \quad k = 0, 1, ..., b-1
$$

그러면 block $b$의 residual risk는:

$$
R_b(m) = \sum_k E_b(k) \cdot p_k^{m}
$$

유도는 residual bit error indicator $z_{b,k}^{(m)} \in {0,1}$를 두면 더 명확하다.

$$
Pr[z_{b,k}^{m} = 1] = p_k^{m}
$$

bit $k$가 residual error로 남았을 때의 damage mass를 $E_b(k)$로 보고 additive expectation을 취하면:

$$
R_b(m)
= E[\sum_k z_{b,k}^{m} \cdot E_b(k)]
= \sum_k E_b(k) \cdot E[z_{b,k}^{m}]
= \sum_k E_b(k) \cdot p_k^{m}
$$

즉 $E_b(k)$는 model-side exposure, $p_k^{m}$는 hardware-side residual failure probability다.

이 식의 장점은 명확하다.

- $E_b(k)$는 모델 구조와 양자화가 결정
- $p_k^{m}$는 ECC 설계가 결정

즉, 모델과 하드웨어를 깔끔하게 분리한 채로 결합할 수 있다.

### Practical Analytic ECC Table Assumption

현재 가장 단순하고 설명 가능한 구현 가정은 다음과 같다.

- codeword 길이 `k`
- ECC correction capability `t`
- raw BER `p`
- codeword 내 bit error는 i.i.d. Bernoulli

이때 codeword 내 raw error 개수 `X`는:

$$
X \sim \text{Binomial}(k, p)
$$

로 둘 수 있다.

그러면 uncorrectable probability는:

$$
UBER(k, t, p) = Pr[X > t]
$$

이고, residual BER는 "복구 실패 후 codeword bit 하나당 기대 residual 오류 수"로 다음처럼 둘 수 있다.

$$
\text{RBER}(k, t, p)
= \frac{1}{k} \sum_{i=t+1}^{k} i \cdot Pr[X = i]
$$

이 정의를 쓰면 ECC mode마다 bit별 residual BER 벡터 $p_k^{m}$를 analytic하게 만들 수 있다.

현재 구현에서는:

- `--ecc_mode name:k:t`
  -> BCH-style cost inference
- `--ecc_mode name:cost:k:t`
  -> manual cost override

의 두 형식을 모두 지원하도록 두는 것이 가장 실용적이다.

여기서 BCH-style parity cost는 단순화하여:

$$
\text{parity}(k, t) = (\log_2 k + 1) \cdot t
$$

로 둔다. 단, 이 규칙은 `k`가 power-of-two일 때만 바로 적용한다.

### 4.1 Baseline Risk

보호가 없는 mode를 $m0$라 하면:

$$
R_b(m0) = \sum_k E_b(k) \cdot p_k^{m0}
$$

이는 위 residual risk 식에 baseline BER 벡터 $p_k^{m0}$를 대입한 특수한 경우다.

### 4.2 Protection Benefit

mode $m0 -> m1$ 전환 시 block $b$의 위험 감소량은:

$$
Benefit_b(m0 -> m1) = R_b(m0) - R_b(m1)
$$

유도라기보다 정의에 가깝다. baseline 대비 보호 적용 후 얼마나 residual risk가 줄었는지를 risk difference로 잡은 것이다.

위 식에 $R_b(m)$ 정의를 대입하면:

$$ \text
{Benefit}_b(m0 -> m1)
= \sum_k E_b(k) \cdot (p_k^{m0} - p_k^{m1})
$$

즉 같은 exposure에 대해 ECC가 BER를 얼마나 낮추는지가 benefit을 결정한다.

이다.

이 값이 실제 budget allocation의 핵심 점수다.

## 5. Optimization Problem

각 block $b$마다 하나의 ECC mode $m_b$를 선택한다고 하자.

- $c(m)$: mode $m$의 parity cost
- $R_b(m)$: block $b$의 residual risk
- $B$: 총 예산

그럼 최적화 문제는:

$$
\begin{aligned}
\min_{m_b}\quad & \sum_b R_b(m_b) \\
\text{subject to}\quad & \sum_b c(m_b) \le B
\end{aligned}
$$

유도는 직접적이다.

- 각 block $b$마다 한 개의 mode만 선택해야 한다.
- 선택 결과의 총 risk는 $\sum_b R_b(m_b)$다.
- 총 parity overhead는 $\sum_b c(m_b)$다.
- 이를 예산 $B$ 이하로 유지하면서 총 risk를 최소화하면 된다.

baseline mode $m0$가 모든 block에 허용된다고 두면, 이 문제는 benefit maximization과 동치다.

$$
R_b(m_b) = R_b(m0) - Benefit_b(m0 -> m_b)
$$

따라서:

$$
\sum_b R_b(m_b)
= \sum_b R_b(m0) - \sum_b Benefit_b(m0 -> m_b)
$$

첫 항은 선택과 무관한 상수이므로, residual risk 최소화는 곧 예산 하 총 benefit 최대화와 같다.

이다.

이는 전형적인 `multiple-choice knapsack` 구조다.

### 5.1 When Greedy Is Enough

다음 조건에서는 greedy가 매우 자연스럽다.

- baseline mode + 보호 mode 1개만 존재
- 모든 block에서 mode cost가 동일

이 경우에는:

$$
Benefit_b = R_b(m0) - R_b(m1)
$$

가 큰 block부터 고르는 것이 최적이다.

추가 비용을 모든 block에서 동일한 $\Delta c$라고 두면, 보호 가능한 block 수는 $K = floor(B / \Delta c)$다. 따라서 문제는

$$
\max \sum_{b \in S} Benefit_b \quad
\text{subject to} \quad |S| <= K
$$

가 되고, 이는 $Benefit_b$가 큰 순서대로 $K$개를 고르는 것이 정확한 최적해다.

cost가 동일하면 `Benefit per cost` 정렬과 $Benefit$ 정렬이 같기 때문이다.

### 5.2 When Knapsack Is Needed

다음 경우에는 knapsack 혹은 그 근사가 필요하다.

- ECC mode가 여러 개 존재
- mode cost가 서로 다름
- block마다 허용 가능한 mode 집합이 다름

이 경우에는 각 block마다 여러 선택지가 있으므로:

$$
\max \text{(total benefit under budget)}
$$

형태의 `multiple-choice knapsack`으로 푸는 것이 맞다.

실무적으로는:

- 작은 후보 집합에서는 DP
- 큰 후보 집합에서는 greedy by marginal benefit/cost 혹은 Lagrangian relaxation

이 현실적이다.

## 6. Sensitive Block Classification Rule

민감 블록 분류는 단순 ranking보다 "보호 우선순위"와 연결되어야 한다.

이를 위해 block별로 다음 두 값을 정의한다.

$$
U_b = R_b(m0) - \min_m R_b(m) \\
\eta_b = \max_m (R_b(m0) - R_b(m)) / c(m)
$$

이 둘 역시 objective에서 바로 나온다.

$$
U_b
= \text{baseline risk - best achievable residual risk}
= R_b(m0) - \min_m R_b(m)
$$

즉 $U_b$는 "이 block에서 아무리 잘 보호해도 줄일 수 있는 위험의 총량"이다.

또한 예산 제약이 있으므로 cost efficiency를 함께 봐야 한다. mode별 protection efficiency를

$$
\eta_b(m) = (R_b(m0) - R_b(m)) / c(m)
$$

로 두면, block $b$의 최선 efficiency는

$$
\eta_b = \max_m \eta_b(m) = \max_m (R_b(m0) - R_b(m)) / c(m)
$$

가 된다.

여기서:

- $U_b$: reducible risk
- $\eta_b$: cost-normalized protection efficiency

### Proposed Classification

다음과 같이 분류한다.

#### Class S: Critical

- $\eta_b$ 상위 블록 중
- 누적 reducible risk의 $80\%$를 덮는 최소 집합

보다 명시적으로는, block을 $\eta_b$ 내림차순으로 정렬해 $b_{1}, b_{2}, ...$라 두고

$$
C_t = \sum_{i=1}^t U_{b_i} / \sum_j U_{b_j}
$$

라 하자. 그러면 Class S는 $C_t >= 0.80$를 처음 만족시키는 최소 prefix다.

이 블록들은 적은 예산으로 큰 위험 감소를 제공하므로 우선 보호 대상이다.

#### Class A: Important

- 그 다음 누적 reducible risk $15\%$를 담당하는 블록

즉 위와 같은 정렬에서 $0.80 < C_t <= 0.95$에 해당하는 다음 구간이다.

이 블록들은 예산이 늘어날 때 두 번째로 보호해야 한다.

#### Class B: Low Priority

- 나머지 블록

즉 $C_t > 0.95$ 이후의 나머지다.

초기 budget 단계에서는 보호할 필요가 없다.

여기서 $80\% / 15\% / 5\%$ 자체는 이론적으로 강제되는 값이 아니라 설명 가능성과 설계 편의성을 위한 policy heuristic이다.

이 분류의 장점은:

- 직관적이고
- risk reduction과 직접 연결되며
- 하드웨어 설계자에게 바로 설명 가능하다는 점이다.

## 7. Adaptive ECC Allocation Algorithm

### Input

1. quantized model checkpoint
2. calibration samples
3. quantization metadata
4. weight-to-codeword packing map
5. ECC mode table:
   - parity cost
   - residual BER vector

실전에서는 이 table을 다음 둘 중 하나로 채우면 된다.

- analytic BCH-style table from `(k, t, raw BER)`
- external hardware-characterized table

### Output

1. block sensitivity score
2. block class ($S/A/B$)
3. block별 ECC mode assignment
4. expected residual risk and expected benefit

### Algorithm

#### Stage 1. Calibration

작은 calibration set으로 activation sensitivity proxy를 수집한다.

예:

$$
h_c \approx (2/N) \cdot \sum_n x_{n,c}^2
$$

이는 calibration sample의 입력 activation 행렬을 $X \in R^{N \times d}$라고 두면, diagonal Gram proxy가

$$
diag((2/N) \cdot X^T X)_c = (2/N) \cdot \sum_n x_{n,c}^2
$$

이기 때문에 나온다. 현재 구현도 각 sample input에 대해 column-wise 제곱합을 누적하는 방식이라 같은 형태다.

이 단계는 training이 아니라 단순 forward statistics 수집이다.

#### Stage 2. Exposure Computation

모든 weight를 한 번 streaming 하면서 $e_w(k)$를 계산한다.

#### Stage 3. Group and Codeword Aggregation

- quantization group exposure $E_g(k)$ 계산
- packing map을 통해 codeword exposure $E_b(k)$ 계산

#### Stage 4. Block Risk Table

모든 block $b$, mode $m$에 대해:

$$
R_b(m) = \sum_k E_b(k) \cdot p_k^m
$$

을 계산한다.

#### Stage 5. Classification

$U_b$와 $\eta_b$를 기준으로 $S/A/B$ class를 만든다.

#### Stage 6. Allocation

예산 $B$ 하에서:

- 단일 mode이면 greedy
- 다중 mode이면 multiple-choice knapsack

으로 최종 ECC assignment를 구한다.

## 8. Why This Is Suitable For 175B

이 방법은 175B에서도 현실적이다.

이유는 다음과 같다.

1. backprop이 필요 없다.
2. repeated fault injection이 필요 없다.
3. training loop가 없다.
4. 계산의 본질이 `one-pass calibration + one-pass aggregation + optimization`이다.

복잡도 관점에서 보면:

- calibration: 소규모 forward statistics 수집
- exposure aggregation: weight 수에 선형
- optimization: 후보 block 수에 대해 수행

즉, 비용은 크지만 training 수준은 아니다.

특히 대형 모델에서는 전체 block을 모두 최적화하지 않고:

- 상위 candidate block만 유지하는 streaming top-k
- block family 단위 coarse-to-fine pruning

을 쓰면 메모리 비용도 낮출 수 있다.

## 9. Recommended Practical Variant

가장 현실적인 실전 버전은 다음과 같다.

### Sensitivity Modeling Unit

`quantization group`

이유:

- shared scale이 있으므로 수식이 단순하다
- 왜 위험한지 설명이 쉽다

### Protection Allocation Unit

`ECC codeword`

이유:

- 실제 하드웨어 보호 단위가 codeword이기 때문이다

### Recommended Flow

$$ \text
{quantization group exposure
-> pack into ECC codeword exposure} \\
\text{-> compute mode-wise residual risk
-> classify blocks
-> allocate ECC budget}
$$

즉, sensitivity는 model-native 단위에서 계산하고, allocation은 hardware-native 단위에서 수행한다.

## 10. Expected Advantages

이 제안의 장점은 다음과 같다.

1. 초대형 모델에도 적용 가능하다.
2. 실험보다 수식 중심이라 설명 가능성이 높다.
3. ECC mode table만 바꾸면 다른 hardware design에도 적용 가능하다.
4. risk와 cost가 직접 연결되므로 실제 예산 배분 문제에 바로 대응한다.

## 11. Main Limitation

이 방법의 핵심 가정은 second-order local proxy가 block sensitivity의 좋은 surrogate라는 점이다.

따라서:

- 절대적인 task degradation 예측
- 복잡한 correlated fault process
- activation fault와 weight fault의 결합

까지 한 번에 모두 설명하는 것은 아니다.

하지만 현재 목표는 "설명 가능한 adaptive ECC allocation"이므로, 이 수준의 abstraction이 가장 합리적이다.

## 12. Bottom Line

제안하는 방법의 핵심은 다음 한 줄로 요약된다.

$$
\text{block sensitivity = model-side exposure} \\
\text{residual risk = exposure} \times \text{ECC residual BER} \\
\text{allocation = minimize residual risk under parity budget}
$$

이 방법은:

- bit ranking 자체를 목표로 하지 않고
- 반복 실험에 의존하지 않으며
- 175B 같은 대형 모델에도 그대로 확장 가능하고
- ECC budget allocation이라는 최종 목적과 직접 연결된다.

따라서 이후 작업은:

1. analytic BCH-style mode set을 실제 hardware 후보군으로 확장
2. packing-aware codeword exposure 계산 정교화
3. residual BER 대신 UBER나 기타 decoder-aware metric이 더 적절한지 검토

에 집중하는 것이 맞다.
