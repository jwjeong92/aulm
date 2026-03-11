# Adaptive Unary Level Mapping Formulation

이 노트는 `proposal.md`를 변경하지 않은 상태에서
현재 `adaptive_unary_level_mapping/` 구현과 일치하도록
제안을 재진술합니다.

## 1. Scope

현재 구현은 다음과 같이 가정합니다:

- unsigned weight quantization (`w_asym=True`)
- representation type in `{int_binary, int_unary, mixed}`
- temporal unary encoding
- temporal unary decoding by nearest Hamming-distance codeword
- tie break toward the smaller decoded integer
- mixed mode with `upos = b(u) = W - u`

Here:

- `W := w_bits`
- `u := u_bits` for one group
- `level_ubits = {u_0 < u_1 < ... < u_{L-1}}` with `u_0 = 0` and `u_{L-1} = W`

`W >= 2`인 경우, 기본 레벨 세트는 다음과 같습니다:

$$
\mathcal U_{\mathrm{default}} = \{0, 2, 3, \dots, W\}
$$

이는 현재 `default_level_ubits()`의 동작과 일치합니다.

## 2. Codec-Level Definitions

Define:

$$
b(u) := W - u
$$

$$
\lambda(u) := 2^u - 1
$$

구현에서 사용하는 인코딩된 너비는 다음과 같습니다:

$$
m(u) =
\begin{cases}
W, & u = 0 \\
\lambda(u) + b(u), & 0 < u < W \\
2^W - 1, & u = W
\end{cases}
$$

이것이 바로 `encoded_width(bits_w, u_bits)` 규칙입니다.

### 2.1 Temporal Unary Encoder

정수 `a in {0, ..., 2^u - 1}`에 대해, Unary 인코더는 다음과 같다:

$$
T_u(a) = 2^a - 1
$$

이것은 소프트웨어와 RTL에서 사용하는 Thermometer 코드입니다.

### 2.2 Temporal Unary Decoder

수신된 unary word `y`에 대해 디코더는 다음을 반환합니다:

$$
\hat a_u(y)
=
\arg\min_{k \in \{0, \dots, 2^u - 1\}}
d_H(y, T_u(k))
$$

여기서 `d_H`는 해밍 거리이다. 여러 `k`가 동일한 최소값을 달성할 경우,
디코더는 더 작은 `k`를 유지한다.

This matches the current nearest-codeword decoder.

## 3. Mixed Unary-Binary Representation

For mixed mode `0 < u < W`, let `q in {0, ..., 2^W - 1}` be the unsigned
quantized weight integer for one weight.\\

> 혼합 모드 `0 < u < W`에 대해, `q in {0, ..., 2^W - 1}`을 하나의 가중치에 대한 부호 없는 양의 양자화 가중치 정수로 정의한다.

Because the current implementation fixes `upos = b(u)`, the unary field
occupies the upper field and the lower `b(u)` bits remain binary. \\
> 현재 구현은 `upos = b(u)`를 고정하므로, unary 필드는 상위 필드를 차지하고 하위 `b(u)` 비트는 binary 상태를 유지합니다.

Define:

$$
a_u(q) := \left\lfloor \frac{q}{2^{b(u)}} \right\rfloor
$$

$$
r_u(q) := q \bmod 2^{b(u)}
$$

Then the mixed encoder is:

$$
C_u(q) = \left(T_u(a_u(q)) \ll b(u)\right) \;|\; r_u(q)
$$

For a corrupted encoded word `y`, the mixed decoder is:

$$
D_u(y) = \left(\hat a_u\!\left(\left\lfloor \frac{y}{2^{b(u)}} \right\rfloor\right) \ll b(u)\right)
\;|\;
\left(y \bmod 2^{b(u)}\right)
$$

Special cases:

$$
C_0(q) = q, \qquad D_0(y) = y
$$

$$
C_W(q) = T_W(q), \qquad
D_W(y) = \arg\min_{k \in \{0, \dots, 2^W - 1\}} d_H(y, T_W(k))
$$

So `int_binary` and `int_unary` are the two endpoint levels of the same
level family.
> 따라서 `int_binary`와 `int_unary`는 동일한 레벨 패밀리의 두 끝점 레벨입니다.

## 4. Representation-Aware Weight Perturbation

The original `proposal.md` uses the binary-only identity:
> 원본 `proposal.md`는 이진 전용 식별자를 사용합니다:

$$
|\Delta q_k| = 2^k
$$

which implies:

$$
(\Delta w_k)^2 = s_g^2 4^k
$$

That identity is exact only for direct binary XOR on the stored weight
integer. It is not exact once the stored representation is unary or mixed,
because the effective integer perturbation is produced by:

> 해당 동일성은 저장된 가중치 정수에 대한 직접 이진 XOR 연산에서만 정확합니다. 저장된 표현이 일진법 또는 혼합일 경우 정확하지 않으며, 이는 유효한 정수 변동이 다음에 의해 생성되기 때문입니다:

1. encode
2. bit-flip injection in encoded space
3. nearest-codeword decode back to integer space

Therefore the correct perturbation unit is not "binary bit position `k`",
but the full encoded-space error pattern.
>따라서 올바른 섭동 단위는 "이진 비트 위치 `k`"가 아니라
전체 인코딩된 공간의 오류 패턴이다.

For a level `u`, let `xi in {0,1}^{m(u)}` be an encoded-space error vector.
Define the decoded integer error:
> Level `u`에 대해, `xi in {0,1}^{m(u)}`를 인코딩된 공간의 오류 벡터로 정의한다.
디코딩된 정수 오류를 다음과 같이 정의한다:

$$
\Delta_u(q, \xi) := D_u(C_u(q) \oplus \xi) - q
$$

Then the local second-order proxy for one weight `w` becomes:
>그러면 한 가중치 `w`에 대한 지역 2차 대리 함수는 다음과 같이 됩니다:

$$
\ell_w(u, \xi)
\approx
\frac{1}{2} \alpha_{\ell(w)} h_{c(w)} s_{g(w)}^2
\left(\Delta_u(q_w, \xi)\right)^2
$$

This replaces the binary-only formula
`(1/2) alpha h s^2 4^k`.
>이것은 이진 전용 공식
`(1/2) alpha h s^2 4^k`를 대체합니다.

## 5. Expected Per-Weight Risk Under BER

Let $\Xi_u(p)$ be the random encoded-space bit-flip vector of length `m(u)`
whose coordinates are i.i.d. Bernoulli with raw BER `p`.
>$\Xi_u(p)$를 길이 `m(u)`의 무작위 인코딩 공간 비트 플립 벡터로 정의하며,
그 좌표들은 독립 동일 분포(i.i.d.) 베르누이 분포를 따르며 원시 BER이 `p`이다.

Then the exact expected risk contribution of one weight under level `u` is:
> 그러면 Level `u`에서 하나의 가중치가 기여하는 정확한 예상 위험은 다음과 같습니다:
$$
r_w(u, p)
:=
\mathbb E_{\Xi_u(p)}
\left[
\frac{1}{2} \alpha_{\ell(w)} h_{c(w)} s_{g(w)}^2
\left(\Delta_u(q_w, \Xi_u(p))\right)^2
\right]
$$

Equivalently:

$$
r_w(u, p)
=
\frac{1}{2} \alpha_{\ell(w)} h_{c(w)} s_{g(w)}^2
\mathbb E_{\Xi_u(p)}
\left[
\left(\Delta_u(q_w, \Xi_u(p))\right)^2
\right]
$$

This is the exact software-consistent formulation for `int_binary`,
`int_unary`, and `mixed`.
> 이것은 `int_binary`, `int_unary` 및 `mixed`에 대한 정확한 소프트웨어 일관성 표현식입니다.

### 5.1 Low-BER Single-Bit Approximation

For small `p`, define the single-bit decoded error energy:

$$
\rho_u(q, t)
:=
\left(D_u(C_u(q) \oplus e_t) - q\right)^2
$$

where `e_t` is the encoded-space unit vector with one flipped bit.

Then:

$$
r_w(u, p)
=
\frac{1}{2} \alpha_{\ell(w)} h_{c(w)} s_{g(w)}^2
\left[
p \sum_{t=0}^{m(u)-1} \rho_u(q_w, t) + O(p^2)
\right]
$$

This approximation makes the implementation difference from `proposal.md`
explicit:

- for `u = 0`, $\rho_0(q, t) = 4^t$, so the old binary formula is recovered
- for `u > 0`, $\rho_u(q, t)$ depends on the value `q`, the unary boundaries,
  and the nearest-Hamming decoder

So unary bit significance is not a function of encoded bit position alone.

### 5.2 Exact Single-Bit Bound for Temporal Unary

The previous subsection gives a low-BER approximation. For the current
temporal unary codec, one can say more for a single encoded-bit flip.

Let:

$$
L := 2^u - 1
$$

and let `e_t` denote a flip of unary-bit position `t in {0, ..., L-1}`,
with LSB indexing. Thus `T_u(a)` has ones exactly on bit positions
`0, 1, ..., a-1`.
For a source integer `a in {0, ..., 2^u - 1}`, define:

$$
\Delta a_u(a, t)
:=
\hat a_u(T_u(a) \oplus e_t) - a
$$

Then the current nearest-Hamming decoder with smaller-`k` tie break satisfies:

$$
\Delta a_u(a, t)
=
\begin{cases}
1, & t = a \\
-1, & t = a - 1 \\
-2, & t = a - 2 \\
0, & \text{otherwise}
\end{cases}
$$

with the understanding that any case with an out-of-range index is ignored.

So for a single unary-bit flip:

$$
\left|\Delta a_u(a, t)\right| \le 2
$$

and this bound is tight.

Proof sketch:

1. `T_u(a)` is a prefix of `a` ones followed by zeros.
2. If `t = a`, the flipped word is exactly `T_u(a+1)`, so `Delta a = +1`.
3. If `t = a-1`, the flipped word is exactly `T_u(a-1)`, so `Delta a = -1`.
4. If `t < a-1`, the received word has one internal hole inside the prefix.
   For `t <= a-3`, keeping decode value `a` is strictly closer than moving the
   boundary down, because lowering the boundary would also mismatch every bit in
   `[t+1, a-1]`. Only when `t = a-2` does `T_u(a-2)` tie with `T_u(a)`, and the
   implementation breaks that tie toward the smaller value, giving `Delta a = -2`.
5. If `t = a+1`, the received word ties between `T_u(a)` and `T_u(a+2)`, so the
   smaller-index tie break keeps the decode at `a`, giving `Delta a = 0`.
6. If `t >= a+2`, the received word is `T_u(a)` plus one isolated `1` farther
   beyond the boundary, and `a` is then the unique nearest codeword, so
   `Delta a = 0`.

For mixed mode, recall that:

$$
q = 2^{b(u)} a_u(q) + r_u(q)
$$

and unary decode errors affect only the unary field. Therefore a single
unary-field flip gives:

$$
\Delta q_u(q, t) = 2^{b(u)} \Delta a_u(a_u(q), t)
$$

hence:

$$
\left|\Delta q_u(q, t)\right| \le 2^{b(u)+1}
$$

and equivalently:

$$
\Delta q_u(q, t)^2 \le 4^{b(u)+1}
$$

This bound is also tight.

Therefore, for one unary-field bit flip, the single-bit error energy
introduced by level `u` is bounded by:

$$
\rho_u(q, t) \le 4^{b(u)+1} = 4^{W-u+1}
$$

### 5.3 Comparison with Binary `4^k`

This exact single-bit result gives a clean comparison to the binary formula.

In mixed mode with level `u`, the unary field replaces the binary bit range:

$$
k \in \{b(u), b(u)+1, \dots, W-1\}
$$

The largest replaced binary single-bit energy is:

$$
\max_{k \in \{b(u), \dots, W-1\}} 4^k = 4^{W-1}
$$

while the unary single-bit worst case is:

$$
\max_t \rho_u(q, t) = 4^{W-u+1}
$$

Hence:

$$
\frac{\max_t \rho_u(q, t)}{4^{W-1}} = 4^{2-u}
$$

So:

- if `u = 2`, the worst unary single-bit error matches the worst replaced
  binary bit
- if `u >= 3`, the worst unary single-bit error is strictly smaller than the
  worst replaced binary bit

However, it is not smaller than every replaced binary bit. The smallest
replaced binary bit energy is:

$$
4^{b(u)}
$$

and:

$$
\max_t \rho_u(q, t) = 4^{b(u)+1} = 4 \cdot 4^{b(u)}
$$

So the correct statement is not
"unary is always smaller than binary `4^k`" but rather:

- unary single-bit errors are capped at `4^{W-u+1}`
- that cap decreases rapidly as `u` increases
- for `u >= 3`, the unary single-bit worst case is already below the binary
  MSB energy of the replaced field
- no such monotone dominance is guaranteed for arbitrary multi-bit error
  patterns `xi`

### 5.4 Exact Double-Bit Bound for Temporal Unary

Now consider exactly two flipped unary bits. Let `xi` satisfy:

$$
|\xi| = 2
$$

and define:

$$
\Delta a_u(a, \xi)
:=
\hat a_u(T_u(a) \oplus \xi) - a
$$

Because `T_u(a)` and `T_u(k)` are thermometer codewords, their mutual
distance is:

$$
d_H(T_u(a), T_u(k)) = |a-k|
$$

Also, `T_u(a) \oplus \xi` is at Hamming distance `2` from `T_u(a)`, so by
triangle inequality:

$$
d_H(T_u(a) \oplus \xi, T_u(k)) \ge |a-k| - 2
$$

The baseline candidate `k = a` has distance exactly `2`. Therefore:

- for a larger candidate `k > a` to beat `a` strictly, it must satisfy

$$
|a-k| - 2 < 2
\quad \Longrightarrow \quad
k-a \le 3
$$

- for a smaller candidate `k < a` to win, tie is already enough because the
  decoder keeps the smaller index under equal distance, so it must satisfy

$$
|a-k| - 2 \le 2
\quad \Longrightarrow \quad
a-k \le 4
$$

Hence the exact decoder output must lie in:

$$
-\min(4, a) \le \Delta a_u(a, \xi) \le \min(3, 2^u - 1 - a)
$$

In particular, the global unary-field double-bit bound is:

$$
-4 \le \Delta a_u(a, \xi) \le 3
$$

so:

$$
\left|\Delta a_u(a, \xi)\right| \le 4
$$

This bound is tight for `u >= 3`, where examples with `Delta a = -4` exist.
For small `u`, the finite alphabet may reduce the achievable maximum.

Proof sketch:

1. The original codeword `T_u(a)` is always a candidate at distance `2`.
2. Any candidate `T_u(k)` farther than `4` below `a` cannot win even under
   favorable triangle-inequality slack, because its distance is still larger
   than or equal to the baseline distance `2`.
3. Any candidate more than `3` above `a` also cannot win, because on the upper
   side the decoder requires strict improvement over `a`.
4. The asymmetry `-4` versus `+3` comes entirely from the smaller-index
   tie-break rule in the decoder.

For mixed mode, a double flip confined to the unary field gives:

$$
\Delta q_u(q, \xi) = 2^{b(u)} \Delta a_u(a_u(q), \xi)
$$

Therefore:

$$
-4 \cdot 2^{b(u)} \le \Delta q_u(q, \xi) \le 3 \cdot 2^{b(u)}
$$

and:

$$
\left|\Delta q_u(q, \xi)\right| \le 4 \cdot 2^{b(u)}
$$

so the squared error satisfies:

$$
\Delta q_u(q, \xi)^2 \le 16 \cdot 4^{b(u)} = 4^{W-u+2}
$$

For `u >= 3`, this worst-case bound is tight. When `u = 2`, the unary alphabet
has only four symbols, so the exact maximum is smaller than the generic upper
bound.

### 5.5 Expected Exposure vs Binary at Low BER

The most natural comparison is not worst-case single-bit energy by itself, but
the expected decoded integer error energy under the same raw BER `p`.

For direct binary storage of a `W`-bit unsigned integer:

$$
\mathbb E[(\Delta q)^2 \mid \text{binary}]
=
p \sum_{k=0}^{W-1} 4^k + O(p^2)
=
p \frac{4^W - 1}{3} + O(p^2)
$$

This is the standard low-BER first-order expansion: each binary bit flips with
probability `p`, and bit `k` contributes `4^k`.

Now consider mixed or unary level `u`, with:

$$
b := W-u,
\qquad
a := a_u(q) = \left\lfloor \frac{q}{2^b} \right\rfloor
$$

The lower `b` bits remain binary, so their first-order contribution is:

$$
\sum_{k=0}^{b-1} 4^k
$$

For the unary field, define:

$$
s_u(a) := \sum_{t=0}^{2^u-2} \frac{\rho_u(q,t)}{4^b}
$$

where `rho_u(q,t)` is the single-bit decoded error energy from Section 5.1.
Because unary-field single-bit errors only change the decoded upper symbol by
`+1`, `-1`, or `-2`, the value of `s_u(a)` is independent of the lower binary
remainder and depends only on `a`.

For `u >= 2`, the exact single-bit rule gives:

$$
s_u(a)=
\begin{cases}
1, & a = 0 \\
2, & a = 1 \\
6, & 2 \le a \le 2^u - 2 \\
5, & a = 2^u - 1
\end{cases}
$$

Therefore the exact first-order mixed/unary exposure is:

$$
\mathbb E[(\Delta q)^2 \mid q, u]
=
p
\left[
\sum_{k=0}^{b-1} 4^k + 4^b s_u(a)
\right]
+ O(p^2)
$$

and since `s_u(a) <= 6`, we always have the uniform upper bound:

$$
\mathbb E[(\Delta q)^2 \mid q, u]
\le
p
\left[
\frac{4^b - 1}{3} + 6 \cdot 4^b
\right]
+ O(p^2)
$$

### 5.6 When Unary Levels Beat Binary on Average

Comparing the previous upper bound to direct binary storage gives a sufficient
condition for unary level `u` to reduce first-order expected exposure for every
source value `q`:

$$
\frac{4^b - 1}{3} + 6 \cdot 4^b
<
\frac{4^W - 1}{3}
$$

Using `W = b + u`, this is equivalent to:

$$
19 < 4^u
$$

Hence:

$$
u \ge 3
\quad \Longrightarrow \quad
\mathbb E[(\Delta q)^2 \mid q, u]
<
\mathbb E[(\Delta q)^2 \mid \text{binary}]
$$

for all `q`, up to first order in `p`.

This is a useful qualitative conclusion:

- unary or mixed coding increases the number of stored bits
- but it sharply reduces the decoded error magnitude of any one encoded-bit
  flip
- for `u >= 3`, that reduction dominates the increased bit count in the
  first-order low-BER regime

The `u = 2` case is borderline and depends on the distribution of the upper
symbol `a`. In that case, binary is beaten whenever the average unary-field
coefficient satisfies:

$$
\bar s < 5
$$

where:

$$
\bar s := \mathbb E[s_u(a)]
$$

For example, if `a` is approximately uniform over `{0,1,2,3}`, then:

$$
\bar s = \frac{1 + 2 + 6 + 5}{4} = 3.5 < 5
$$

so even `u = 2` is favorable on average under that distribution.

### 5.7 Limitation of the First-Order Comparison

The comparison above is a low-BER first-order statement. It neglects the
`O(p^2)` multi-bit events.

This matters because unary levels have encoded width:

$$
m(u) = (2^u - 1) + (W-u)
$$

for mixed mode, and `m(W) = 2^W - 1` for full unary. As `u` grows, the number
of possible double-bit and higher-order error patterns grows rapidly.

So the first-order conclusion should be interpreted as:

- reliable for sufficiently small BER
- directionally correct in the low-BER regime
- not a replacement for explicit higher-order analysis or empirical BER sweep
  when `p` is large enough that multi-bit events are non-negligible

### 5.8 Comparison Intuition by Level

The first-order comparison can be summarized as a tradeoff between:

- more encoded bits, which creates more opportunities for raw flips
- smaller decoded integer jumps, which reduces the damage per flip

For mixed level `u`, the lower binary field still behaves like ordinary binary,
while the upper `u`-bit field is replaced by a unary field of length `2^u-1`.
In the low-BER regime, only a small number of unary bits near the thermometer
boundary contribute nonzero first-order decoded error, which is why unary can
still win despite having more stored bits.

For the common `W = 6` setting used in this repository:

| level `u` | binary bits kept `b=6-u` | mixed encoded width `m(u)` | unary-field worst single-bit Δq (abs.) | unary-field worst double-bit Δq (abs.) | first-order vs binary |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 6 | 6 | not applicable | not applicable | baseline binary |
| 2 | 4 | 7 | 32 | 48 | borderline; distribution-dependent |
| 3 | 3 | 10 | 16 | 32 | improved in first-order expectation |
| 4 | 2 | 17 | 8 | 16 | improved; lower jump size dominates |
| 5 | 1 | 32 | 4 | 8 | improved at low BER, but higher-order events grow |
| 6 | 0 | 63 | 2 | 4 | smallest jump size, largest encoded width |

This table makes the overall trend explicit:

- increasing `u` shrinks the decoded jump size exponentially
- increasing `u` also expands encoded width exponentially
- the single-bit first-order regime favors larger `u`
- the multi-bit regime eventually pushes back as `m(u)` becomes large

So, qualitatively:

- `u = 2` is a cautious mixed point with small storage overhead and modest
  robustness gain
- `u = 3` is the first level with a uniform first-order advantage over binary
- `u = 4` is often the practical sweet spot because it materially reduces
  decoded error while keeping encoded width manageable
- `u = 5` and `u = 6` suppress single-bit damage strongly, but they pay a much
  larger width penalty and therefore become more sensitive to higher-order BER
  effects

This is exactly why the current adaptive search does not force all groups to
full unary. Instead it searches over the ordered level set `level_ubits` and
lets low-importance groups absorb more unary expansion while preserving
accuracy and overhead constraints.

## 6. Group-Level Risk

For a quantization group `g`, the exact group risk at level `u` is:

$$
R_g(u, p) = \sum_{w \in g} r_w(u, p)
$$

If a layer-wide BER sweep `mathcal B` is used, one can aggregate over BER
points as:

$$
R_g(u; \mathcal B) = \sum_{\beta \in \mathcal B} \omega_{\beta} R_g(u, \beta)
$$

for some optional nonnegative weights `omega_beta`.

The current search code does not directly optimize this analytic sum.
Instead it uses the level assignment as a candidate policy, injects errors,
runs `lm_eval`, and applies pass/fail constraints per task and per BER point.
Still, `R_g(u, p)` is the correct analytic quantity if we want a formula that
matches the codec.

## 7. Level Policy from `level_ubits`

Let the total number of quantization groups be `G`, and let

$$
I_g
$$

be the importance score of group `g` from the selected importance metric.

Sort groups in ascending importance:

$$
\pi(1), \pi(2), \dots, \pi(G)
$$

so that:

$$
I_{\pi(1)} \le I_{\pi(2)} \le \dots \le I_{\pi(G)}
$$

Let:

$$
\mathcal U = \{u_0, u_1, \dots, u_{L-1}\}
$$

and let the adaptive search choose nondecreasing level boundaries:

$$
0 \le n_0 \le n_1 \le \dots \le n_{L-2} \le G
$$

Then the group-level policy is:

$$
u_{\pi(r)} =
\begin{cases}
u_0, & 1 \le r \le n_0 \\
u_{\ell}, & n_{\ell-1} < r \le n_{\ell}, \quad \ell = 1, \dots, L-2 \\
u_{L-1}, & n_{L-2} < r \le G
\end{cases}
$$

This matches the current `build_group_policy()` logic:

- all groups start at the last level `u_{L-1} = W`
- progressively more low-importance groups are moved to smaller `u`
- the frozen boundaries define the final per-group `u_bits` policy

## 8. Storage Overhead

If group `g` contains `n_g` weights and is assigned level `u_g`, the exact
encoded storage overhead ratio is:

$$
\Omega(\{u_g\})
=
\frac{\sum_g n_g \, m(u_g)}
{\sum_g n_g \, W}
$$

When all groups have the same size, this simplifies to:

$$
\Omega(\{u_g\})
=
\frac{1}{G W} \sum_{g=1}^{G} m(u_g)
$$

This is exactly the quantity reported by `compute_policy_overhead()`.

## 9. Search Constraint and Tie-Break Rule

Let `A_t^{base}` be the clean quantized baseline accuracy for task `t`, and
let `bar{A}_t({u_g}; beta)` be the mean accuracy over seeds for candidate
policy `{u_g}` at BER `beta`.

With tolerance `tau`, a candidate passes iff:

$$
\bar{A}_t(\{u_g\}; \beta) \ge (1 - \tau) A_t^{base}
\qquad
\text{for all tasks } t \text{ and BER points } \beta
$$

Define the minimum margin:

$$
M(\{u_g\})
:=
\min_{t, \beta}
\left[
\bar{A}_t(\{u_g\}; \beta) - (1 - \tau) A_t^{base}
\right]
$$

The implemented search rule is:

1. prefer passing candidates over failing candidates
2. among passing candidates, minimize `Omega`
3. tie break by maximizing `M`

So the actual optimization variable is not an ECC mode assignment but the
boundary vector:

$$
(n_0, n_1, \dots, n_{L-2})
$$

over the ordered level set `level_ubits`.

## 10. Main Correction Relative to `proposal.md`

The key correction is:

$$
\text{binary-only } 4^k
\quad \longrightarrow \quad
\text{codec-aware } \left(D_u(C_u(q) \oplus \xi) - q\right)^2
$$

As a result:

- the perturbation model is level-dependent
- unary and mixed representations are decoder-dependent
- bit significance depends on both encoded bit location and the source value
  `q`
- `level_ubits` becomes a first-class optimization object
- overhead is exponential in `u` through `m(u) = (2^u - 1) + (W - u)` for
  mixed levels

This is the formulation that should be used for any document meant to match
the current `adaptive_unary_level_mapping/` code path.
