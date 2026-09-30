# futhark-lab

Futhark experiments for the fleet's cellular substrate, with the compiler constraints
written down so the next person does not rediscover them by failure.

## What is here

| file | state | what |
|---|---|---|
| `BRIEF.md` | — | **start here.** 12 compiler constraints, 8 experiments ordered by information/hour |
| `exp1_tick.fut` | **compiles** | TICK as one kernel over the whole quilt (padded adjacency) |
| `exp1c_tick_csr.fut` | **does not compile — on purpose** | the same kernel in CSR, kept as the failing control |
| `exp5_witness_chain.fut` | **compiles** | the witness chain as a prefix problem |
| `canary.fut` | **compiles** | FNV-1a 64 canary, `0x24a555471370b18d` |
| `BQN-FINDINGS.md` | — | the sibling port, blocked at the fold-seeding stage |

## The one-paragraph version

A cell's dial movement depends only on its graph neighbours, so the whole graph should
tick as one array operation rather than N method calls. That kernel compiles
(`exp1_tick.fut`). The right data layout for it does not: CSR is O(V+E) while padding is
O(V·d_max), a difference of up to 217x on a real graph, and the CSR form does not typecheck
because Futhark cannot carry a per-row shape through a gather. **The bet is blocked by the
type system, not the GPU.** That is the most useful thing this lab has produced, and the
two programs above are the evidence for it.

## Ground rules

Any number that has not been through `0x24a555471370b18d` is not a measurement. Two
verification bugs in this lab were both in the *check*, not the code — a mistyped decimal
for a hex digest, and a `tr -d 'u64'` that deleted the `6` and `4` out of the number
itself. The check is where the errors live. Instrument it.
