연구 배경:
- LLM 모델 크기 증가로 인해 weight storage capacity 요구사항이 빠르게 증가하고 있음
- 이를 완화하기 위해 Flash memory를 활용한 LLM serving system이 제안되고 있음
- 이러한 IFP system의 핵심은 대규모 weight를 Flash array 근처에 저장하고, chip 내부에서 GEMV를 수행해 data movement를 줄이는 것임
- 다만 기존 연구들은 대체로 harsh TLC read-disturb 환경을 직접 다루기보다, 더 완화된 신뢰성 조건을 가정함
  - Lincoln은 SLC mode + BCH 기반 보호를 사용함
  - AiF는 TLC cell을 사용하더라도 weight를 LSB page에만 저장하고 BCH 기반 ECC를 사용함
  - Cambricon-LLM은 TLC를 대상으로 하지만, 허용 error rate를 대체로 `1e-4` 이하의 낮은 영역으로 제한함

문제점:
- TLC 이상의 NAND Flash memory에서는 raw bit error와 read-disturb 누적 효과가 더 커지므로, SLC/LSB-page 기반 설계보다 더 강한 보호가 필요함
- 그러나 stronger ECC를 그대로 적용하면, Flash-based IFP가 추구하는 capacity/throughput/logic-efficiency 이점이 크게 줄어들 수 있음
  - 본 노트에서는 ECC 후보군을 `BCH(1KiB codeword, k=8192b)`의 `10t`, `50t`, `64t`로 둠
  - 이때 BCH parity bit는 각각 `140b`, `700b`, `896b`이며, 순수 parity overhead는 각각 약 `1.7%`, `8.5%`, `10.9%`임
  - 그러나 TLC에서 LSB page만 사용하는 방식은 page utilization이 본질적으로 `1/3` 수준이므로, parity 자체보다 page utilization 손실이 capacity 측면에서 더 지배적일 수 있음
  - 즉 LSB-only TLC + BCH의 effective user-data ratio는 raw TLC 대비 대략 `32.8%`(`10t`), `30.7%`(`50t`), `30.0%`(`64t`) 수준에 머무름
  - 또한 AiF의 Figure 5는 `50-bit / 1-KiB` BCH decoder 기준으로 `102.4 GB/s`를 달성하려면 누적 on-chip ECC decoder가 약 `40.12 mm^2`의 silicon area와 `10.694 W`의 전력을 요구한다고 보고함
  - 이는 consumer-grade SSD의 전형적인 power budget(`6-8 W`)을 크게 초과하는 수치이므로, naive TLC + strong ECC가 SSD-level PPA 제약과 양립하기 어렵다는 점을 직접적으로 보여줌
  - 따라서 page utilization, ECC parity overhead, decoder area/power, decode latency를 함께 고려하면 naive TLC + strong ECC는 비효율적일 수 있음
- 반대로 ECC 강도를 낮추면, 시스템은 read-reclaim과 read-retry 사이의 tradeoff를 더 직접적으로 감당해야 함
  - Read-reclaim은 block read count가 threshold에 도달했을 때 block migration을 수행해 read-disturb 누적을 완화하는 동작임
  - LLM decoding에서는 모델 weight가 한 token마다 반복적으로 읽히므로, hot block의 read count는 token 수에 거의 선형으로 증가함
  - 본 노트에서는 SSD organization을 `16 channel x 4 chip x 4 plane = 256-way parallelism`, block size를 `768 pages/block`으로 가정함
  - 이때 int8 quantized `7B/30B/70B/100B` 모델은 모두 realistic NAND page size 범위에서 `1 block x all parallelism`보다 큼
  - 구체적으로 int8 7B 모델조차 parallel unit당 1 block 이상을 차지하려면 page size가 `34.8 KiB` 이하여야 하는데, 이는 일반적인 NAND page size 범위보다 큼
  - 따라서 practical setting에서는 `7B` 이상 모델 모두 대부분의 weight block이 지속적으로 재읽히는 hot block이 된다고 볼 수 있음
  - 따라서 reclaim threshold가 낮고 throughput이 높을수록 migration traffic이 빠르게 누적되며, 이는 endurance budget을 빠르게 소모할 수 있음
  - 예를 들어 read-reclaim threshold가 `200K`, block endurance budget이 `3K P/E`이면, hot block 하나가 감당할 수 있는 총 누적 read는 약 `6e8`회임
  - 위와 같은 hot-block full-sweep 가정에서는 token당 hot block read count가 약 `768` 증가하므로, hot block의 lifetime token budget은 `6e8 / 768 ≈ 7.8e5 tokens`에 불과함
  - 목표 서비스 조건을 `5 tok/s`, `3년`으로 두면 총 요구 token 수는 약 `4.73e8 tokens`이므로, 이는 hot-block lifetime budget보다 약 `605x` 큼
  - 같은 조건을 뒤집어 보면, `200K` reclaim threshold와 `3K P/E` budget으로 3년을 버티기 위해서는 hot block당 평균 read count 증가가 token당 약 `1.27 page` 이하여야 함
  - 즉 full-block sweep에 가까운 LLM serving access pattern에서는 read-reclaim 부담이 구조적으로 매우 커질 수 있음
  - Read-reclaim을 늦추면 migration write는 줄일 수 있지만, 그만큼 read-disturb 누적과 ECC failure 가능성은 커짐
  - Read-retry는 ECC decoding 실패 후 read reference voltage를 바꿔 재읽는 기법이며, 실패 빈도가 늘수록 read latency와 effective bandwidth가 감소함
  - `refs/motivation.pdf` slide 7의 모델에서는 `tR = 30 us`, `tECC = 561 ns`, `N_RR = 1/3/5`로 두고 있으며, read-retry 1회 비용은 ECC decode 자체보다 약 `53.5x` 큼
  - 같은 모델에서 access당 read path time은 대략 `30.56 us`, `90.56 us`, `150.56 us` 수준이므로, retry 횟수가 늘수록 latency와 per-plane effective bandwidth가 거의 retry count에 비례해 악화됨
  - 즉 harsh TLC 조건에서는 수명 보호를 위해 reclaim을 자주 수행하거나, 성능 보호를 위해 retry를 자주 수행하는 딜레마가 발생함

기회:
- LLM은 bit error에 대해 균일하게 민감하지 않으며, 일부 작은 오류나 비민감 weight에서의 오류는 최종 accuracy degradation이 제한적일 수 있음
- 따라서 ECC가 failure를 일으켰더라도, 모든 residual error에 대해 동일하게 read-retry를 수행할 필요는 없을 수 있음
- Thermometer code라고도 불리는 temporal unary code는 binary 표현보다 작은 decoded perturbation을 유도할 수 있어 low-BER 영역에서 오류 완화 효과를 제공할 수 있음
- 특히 unary/mixed representation은 single-bit error가 만드는 decoded integer jump를 줄여, weight perturbation magnitude를 낮출 수 있음
- 다만 unary code는 encoded width가 증가하므로 storage overhead와 higher-order multi-bit event 노출이 함께 증가함
- 따라서 full unary를 일괄 적용하는 것이 아니라, accuracy gain과 space overhead를 함께 고려한 selective/adaptive unary mapping이 필요함
- 본 연구에서는 특히 `BER = 1e-2` 수준에서도 accuracy drop을 `5%` 이내로 유지하는 것을 목표 조건으로 둠

본 연구의 목적은 harsh TLC Flash 환경에서, read-retry/read-reclaim 부담을 줄이면서도 LLM accuracy를 유지할 수 있는 unary-based encoding policy를 탐색하는 것이다.
