# Paper Plan

Date: 2026-03-11
Workspace: `/home/jwjeong/study/quant_analysis`

## Purpose

이 문서는 `overview.md`를 논문으로 발전시키기 위한 실행 계획의 기준 문서다.
다음 세션은 이 파일을 먼저 읽고, 현재 단계와 다음 작업을 바로 이어서 수행해야 한다.

## Source of Truth

- `paper_plan.md`
  - 현재 작업 단계, 다음 우선순위, 세션 운영 규칙
- `paper_session_bridge.md`
  - 배경, 고정 가정, 이미 계산된 수치, 참고 문헌 anchor
- `paper_search_strategy_note.md`
  - 현재 search redesign 결정과 다음 구현 경로
- `overview.md`
  - 현재까지 정리된 motivation 초안

## Paper Thesis

논문의 중심 주장은 다음이다.

> 기존 flash-based LLM inference 연구는 SLC/LSB-page/low-BER 조건으로 harsh TLC 문제를 완화해 왔다.
> 그러나 실제 harsh TLC에서는 strong ECC가 SSD-level PPA 측면에서 부담이 크고, weak ECC는
> read-retry/read-reclaim 부담을 키운다. 본 연구는 LLM의 비균일한 오류 민감도를 활용해
> unary-based encoding policy로 이 tradeoff를 완화하려 한다.

## Non-Negotiable Scope

- weight-only quantized LLM inference
- flash-resident 또는 flash-near execution
- harsh TLC reliability regime 중심
- unary / mixed-unary 기반 완화 기법이 논문 중심
- generic ECC allocation paper로 흐르지 말 것
- 대규모 반복 fault injection이 논문 중심이 되지 말 것

## Locked Assumptions

다음 조건은 사용자가 이미 확정한 값으로 간주한다.

- SSD organization: `16 channel x 4 chip x 4 plane = 256-way parallelism`
- block size: `768 pages/block`
- throughput target: `5 tok/s`
- service lifetime target: `3 years`
- reclaim threshold: `200K`
- endurance budget: `3K P/E`
- models: int8 `7B / 30B / 70B / 100B`
- ECC family: `BCH(1KiB codeword, k=8192 bits)` with `10t / 50t / 64t`
- accuracy target: `BER = 1e-2`에서 accuracy drop `<= 5%`
- placement: maximum-bandwidth striped/distributed layout

세부 수치와 해석은 `paper_session_bridge.md`를 따른다.

## What Is Already Safe To Claim

- 기존 flash-based LLM 연구는 harsh TLC를 직접 다루기보다 완화된 operating point에 머문다.
- naive strong ECC는 capacity, area, power 측면에서 SSD-level 제약과 충돌할 수 있다.
- large-model flash inference에서는 retry/reclaim 부담이 구조적으로 커질 수 있다.
- unary / mixed unary는 single-bit decoded perturbation을 줄일 수 있다.

## What Must Still Be Proven

- selective/adaptive unary가 `BER = 1e-2`에서 `<= 5%` accuracy drop을 달성하는가
- unary-based policy가 binary baseline 대비 충분히 유의미한 robustness-overhead 이득을 보이는가
- aggressive fixed unary expansion보다 더 실용적인 sweet spot policy가 존재하는가

이 세 가지는 결과가 확보되기 전까지 논문 본문에서 확정형으로 쓰지 않는다.

## Required Paper Deliverables

논문 작성 과정에서 아래 문서들을 순서대로 만들고 유지한다.

1. `paper_outline.md`
   - 제목 후보, one-sentence thesis, contribution bullets, section outline
2. `paper_method.md`
   - `proposal.md`와 `proposal_adaptive_unary_level_mapping.md`를 통합한 canonical method note
   - 동시에 아직 비어 있는 evidence / figure / table / equation inventory를 명시
3. `paper_experiment_matrix.md`
   - 모델, BER point, baseline, metric, pass/fail condition
4. `paper_draft.md`
   - 실제 논문 텍스트 초안

## Current Phase

- current phase: `Phase 4. Evidence Closure`
- completed this session:
  `paper_search_strategy_note.md`
- next deliverable:
  surrogate-guided search prototype enabling Phase 4 execution
- execution status:
  Phase 4 evidence runs are temporarily paused until search redesign is frozen

## Execution Plan

### Phase 1. Story Freeze

Goal:
- 논문 중심축을 고정한다.

Deliverable:
- `paper_outline.md`

Must contain:
- provisional title candidates
- one-sentence thesis
- 3-4 contribution bullets
- section-by-section outline

Done criteria:
- `overview.md`의 motivation을 다시 뒤엎지 않아도 되는 상태
- unary paper인지, ECC allocation supporting role인지 명확한 상태

### Phase 2. Method Consolidation

Goal:
- 현재 흩어진 방법론 문서를 논문용 언어로 정리한다.

Deliverable:
- `paper_method.md`

Must do:
- `proposal.md`의 analytic risk 구조와
  `proposal_adaptive_unary_level_mapping.md`의 representation-aware unary math를 분리해서 정리
- binary-only 식을 unary/mixed case에 그대로 쓰지 않도록 명시
- notation을 논문용으로 정리

Done criteria:
- 방법 섹션을 한 파일만 읽고도 작성 가능한 상태

### Phase 3. Evaluation Matrix Freeze

Goal:
- 실험 범위와 성공 기준을 고정한다.

Deliverable:
- `paper_experiment_matrix.md`

Must specify:
- 사용할 모델
- BER points
- baseline 목록
- metric 목록
- `<= 5%` 목표를 어떤 benchmark / score로 판정할지
- overhead를 어떤 숫자로 보고할지

Done criteria:
- 추가적인 실험 기획 논쟁 없이 실행 가능한 상태

### Phase 4. Evidence Closure

Goal:
- 논문 핵심 주장에 필요한 최소 증거를 확보한다.

Must answer:
- `BER = 1e-2`에서 confirmatory baseline family
  (`binary / fixed mixed / adaptive`) 중 무엇이 가장 좋은가
- best policy의 encoded-width overhead는 얼마인가
- best policy가 heavier unary expansion보다 실용적인가
- retry/reclaim pressure 완화라는 해석을 붙일 만큼 결과가 충분한가

Done criteria:
- 논문 claims 중 "must still be proven" 항목에 대해 근거가 생긴 상태

### Phase 5. Draft Writing

Goal:
- 실제 논문 초안을 작성한다.

Deliverable:
- `paper_draft.md`

Recommended order:
1. Introduction
2. Background / Motivation
3. Problem Definition
4. Method
5. Experimental Setup
6. Results
7. Related Work
8. Discussion / Limitations
9. Conclusion

Done criteria:
- 전체 논문이 end-to-end로 읽히는 상태

### Phase 6. Final Tightening

Goal:
- 과한 주장, 약한 연결, 중복 서술을 제거한다.

Checklist:
- claim과 evidence가 1:1로 대응하는가
- motivation 수치가 method / result와 연결되는가
- unary의 장점과 한계가 모두 명시되는가
- "harsh TLC"가 단순 배경이 아니라 결과 해석까지 연결되는가

## Immediate Next Session Tasks

다음 세션은 아래 순서로 진행한다.

1. `paper_plan.md` 읽기
2. `paper_session_bridge.md` 읽기
3. `paper_search_strategy_note.md` 읽기
4. `paper_outline.md` 읽기
5. `paper_method.md` 읽기
6. `paper_experiment_matrix.md` 읽기
7. `proposal_adaptive_unary_level_mapping.md` 필요 시만 재확인
8. `proposal.md` 필요 시만 재확인
9. `overview.md` 필요 시만 재확인
10. surrogate-guided search prototype 설계 및 구현 시작

다음 세션의 첫 작업은 반드시 search redesign 구현 시작이다.
`E1-E6` 실행은 새 search path가 준비된 뒤 재개한다.

## Session Operating Rules

세션 시작 시:

- 현재 phase를 먼저 확인한다.
- 그 phase의 deliverable 하나만 만든다.
- motivation 자체를 다시 확장하지 않는다. 모순이 있을 때만 수정한다.
- 데이터가 아직 비어 있는 문서는 필요한 evidence / figure / table /
  equation inventory를 반드시 함께 남긴다.

세션 종료 시:

- `paper_plan.md`의 현재 상태를 업데이트한다.
- `paper_session_bridge.md`에 새로 확정된 수치나 의사결정을 반영한다.
- 다음 세션의 first task를 한 줄로 남긴다.

## Current Status

현재 상태는 다음과 같다.

- motivation 초안은 `overview.md`에 정리되어 있음
- 배경, 가정, 수치, reference anchor는 `paper_session_bridge.md`에 정리되어 있음
- `paper_outline.md`가 작성되어 story freeze가 끝남
- `paper_method.md`가 작성되어 canonical method language가 정리됨
- `paper_experiment_matrix.md`가 작성되어 exact experiment scope가 고정됨
- `paper_search_strategy_note.md`가 작성되어 search redesign 방향이 고정됨
- working title은 현재
  `Adaptive Unary Encoding for Harsh-TLC Flash-Based LLM Inference`로 둠
- pilot matrix는 현재 `RTN`, `w_bits = 6`, `OPT-125M`,
  `piqa/hellaswag/arc_easy`, BER
  `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`로 고정됨
- main confirmatory matrix는 현재 `RTN`, `w_bits = 8`, `OPT-6.7B`,
  `piqa/hellaswag/arc_easy`, BER
  `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`로 고정됨
- `7B / 30B / 70B / 100B` int8은 system projection track으로 유지됨
- `6-bit`는 pilot / ablation이고, paper-critical empirical evidence는
  `int8` confirmatory track에 둠
- 현재 global empirical greedy search는 너무 느려서 main DSE path로는
  쓰지 않기로 함
- 다음 작업은 surrogate-guided search prototype 구현임
- `paper_method.md` 안에 실험으로 채워야 할 data / figure / table /
  equation inventory가 포함되어 있음

## Open Decisions For Later

지금 바로 결정하지 말고, 필요할 때만 다시 연다.

- target venue style
- 본문에서 ECC allocation을 얼마나 비중 있게 다룰지
- 결과 그래프의 주 metric을 accuracy 단독으로 둘지, joint tradeoff plot로 갈지

## Read Order For Any Future Session

문맥이 부족한 세션은 반드시 아래 순서로 읽는다.

1. `paper_plan.md`
2. `paper_session_bridge.md`
3. `paper_search_strategy_note.md`
4. `paper_outline.md`
5. `paper_method.md`
6. `paper_experiment_matrix.md`
7. `proposal_adaptive_unary_level_mapping.md`
8. `proposal.md`
9. `overview.md`

이 순서를 따르면 바로 다음 작업을 이어갈 수 있다.
