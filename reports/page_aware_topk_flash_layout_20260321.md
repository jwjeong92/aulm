# Hardware-Friendly Protection Selection and Flash Layout

## Observation

The current top-5 selected-group plots suggest that selection is not uniformly distributed across the weight tensor; instead, it is strongly concentrated on specific input-column bands. This indicates that the importance structure is at least partly column-dependent rather than purely random at the per-group level.

## Implication

If selected groups receive a higher protection level, they will incur higher redundancy, while unselected groups will remain at a lower redundancy level. Under this condition, a naive variable-length row-major bitstream layout is undesirable for flash storage because it increases fragmentation, complicates address computation, and breaks alignment with flash transaction units.

## Recommended Storage Layout

A better approach is to organize the weight tensor using a column-group-major tiled layout. Let the logical storage tile be

\[
\tau = (\ell, c, r)
\]

where:
- \(\ell\): layer index
- \(c\): column-group index
- \(r\): row-tile index

Here, one column-group corresponds to the current quantization/protection group size, e.g. 128 weights along the input-column dimension.

The physical flash mapping should use:
- `4 KiB` as the transaction-aware unit (`sector` or `subpage`)
- `16 KiB` as the page unit
- `1 page = 4 sectors`

The recommended policy is:
- use `sector` as the selection granularity
- use `page` as the packing and metadata granularity
- preserve column-group locality across rows
- avoid globally interleaving groups with different protection widths

## Page / Sector Organization

Each `16 KiB` page stores tiles that belong to the same or nearby column-groups. A page contains up to four `4 KiB` sectors. Each sector stores a set of row-tile groups with similar protection characteristics.

A practical page layout is:
- page header
- sector 0 payload
- sector 1 payload
- sector 2 payload
- sector 3 payload

The page header stores:
- layer id
- column-group id
- row-tile range
- sector offsets
- protection-class map
- optional local decoding metadata

This makes addressing and decoding simpler while preserving the observed column-local structure.

## Protection-Aware Capacity Numbers

For `w_bits = 6` and `group_size = 128`, the encoded payload per group is approximately:

| mode | encoded width | bytes per group |
| --- | ---: | ---: |
| binary | 6b | 96 B |
| mixed `u=4` | 17b | 272 B |
| mixed `u=5` | 32b | 512 B |
| full unary `u=6` | 63b | 1008 B |

Approximate packing density:

| mode | groups per 4 KiB sector | groups per 16 KiB page |
| --- | ---: | ---: |
| binary | 42 | 170 |
| mixed `u=4` | 15 | 60 |
| mixed `u=5` | 8 | 32 |
| full unary `u=6` | 4 | 16 |

These numbers show why direct per-group variable-length placement is inefficient: the storage footprint changes too sharply with protection level.

## Selection Approximation

The ideal formulation is exact per-group top-k protection. However, that choice is not flash-friendly because the physical transaction unit is not the group itself but the sector/page.

Therefore, a hardware-friendly approximation is to lift the decision unit from `group` to `sector-aligned tile`, while preserving as much column locality as possible.

## Sector-Aware Top-k Formulation

For each sector-aligned tile \(\tau\) and protection mode \(m\), define:
- \(B_{\tau,m}\): aggregated protection benefit
- \(C_{\tau,m}\): additional storage cost measured in sectors

A natural definition is:

\[
B_{\tau,m} = \sum_{g \in \tau} b_{g,m}
\]

\[
C_{\tau,m} =
\left\lceil \frac{\mathrm{bytes}(\tau,m)}{4\mathrm{KiB}} \right\rceil
-
\left\lceil \frac{\mathrm{bytes}(\tau,0)}{4\mathrm{KiB}} \right\rceil
\]

Then the sector-aware budgeted top-k problem is:

\[
\max \sum_{\tau} \sum_m x_{\tau,m} B_{\tau,m}
\]

subject to

\[
\sum_{\tau} \sum_m x_{\tau,m} C_{\tau,m} \le B_{\mathrm{sec}}
\]

\[
\sum_m x_{\tau,m} \le 1
\]

\[
x_{\tau,m} \in \{0,1\}
\]

This is the most practical formulation when flash transactions are sector-based.

## Page-Aware Regularization

Since sectors are packed into `16 KiB` pages, the selection objective should also discourage excessive page fragmentation. This can be done by adding a page-level penalty:

\[
\max \sum_{\tau} \sum_m x_{\tau,m} B_{\tau,m}
- \lambda \sum_p \mathrm{frag}(p)
\]

where \(\mathrm{frag}(p)\) measures how mixed the protection classes are inside page \(p\).

This gives a two-level policy:
- sector-aware selection for transaction efficiency
- page-aware regularization for layout simplicity and better flash management

## Approximation Hierarchy

In terms of resiliency:

\[
\mathrm{exact\ group\ top\mbox{-}k}
>
\mathrm{sector\mbox{-}aware\ selection}
>
\mathrm{page\mbox{-}aware\ selection}
\]

In terms of hardware / flash friendliness:

\[
\mathrm{page\mbox{-}aware\ selection}
>
\mathrm{sector\mbox{-}aware\ selection}
>
\mathrm{exact\ group\ top\mbox{-}k}
\]

Therefore, the best practical compromise is:

**sector-aware selection with page-aware packing and fragmentation control.**

## Practical Recommendation

The recommended implementation is:
- define each decision tile as `(layer, column-group, row-tile)`
- aggregate group-level importance/benefit into `4 KiB` sector candidates
- select sectors under a global storage budget
- pack four sectors into each `16 KiB` page
- penalize mixed-protection pages if needed

This preserves the observed column-local importance pattern, aligns protection assignment with actual flash transaction granularity, and avoids the large fragmentation overhead of exact per-group selection.
