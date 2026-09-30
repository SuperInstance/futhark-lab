# Futhark lab — edge-finding brief for the GPU agent

Futhark **0.27.1**, verified by execution on this machine. Two programs compile and are
checked in alongside this file. Everything marked FOUND was discovered by hitting it, not
by reading the manual — the manual does not state most of these.

---

## PART 1 — Compiler constraints, so you do not burn an hour on each

### 1.1 There is no elementwise arithmetic on 2-D arrays. None.

```futhark
def main [n][m] (a: [n][m]f64) (b: [n][m]f64) : [n][m]f64 = a + b
```
→ `Cannot apply "+" to "a" (invalid type). Expected: [][]f64. Actual: [][]f64`

They print as the same type. They are not. Arithmetic in Futhark is **1-D only**.

Two working idioms:

```futhark
-- (a) fully nested map -- always works
map (\i -> map (\j -> a[i][j] + b[i][j]) (iota m)) (iota n)

-- (b) row-wise map2, usually shorter
map2 (\ra rb -> map2 (+) ra rb) a b
```

`map2 (+) a b` on 2-D **also fails** — `map2` on a 2-D array maps over *rows*, so `+` is
being handed two rows when it wants scalars. The same is true of `map`.

This is the single most likely thing to waste an hour. Assume every multi-dimensional
program needs a nested map.

### 1.2 A gather and an arithmetic op cannot share a lambda

```futhark
map (\j -> dials[adj[i][j]] - dials[i])     -- does not typecheck, shape will not infer
map (\j -> dials[adj[i][j]])                 -- fine on its own
```

Split into passes, or accumulate over the index axis with `foldl`. `exp1_tick.fut` shows
the working form.

### 1.3 Size parameters `[n][d]` are entry-point-only

On an internal `def` they are *"Unknown name"*. Use an unsized parameter (`[]u64`) for
internal helpers, or inline them.

### 1.4 Reserved words that will cost you

`type`, `entry`, `entrypoint`. All three. The entry point is `def main`.

### 1.5 No implicit numeric promotion, and no integer XOR

`u8` does not widen to `u64`, not even by adding a `u64` literal. Carry the wider type
from the input. There is no integer XOR; it is `(a|b) - (a&b)`, which is exact.

`u64` **wraps natively**, which is exactly what FNV-1a wants. This is Futhark's single
biggest advantage over BQN, where a 64-bit value cannot be represented at all.

### 1.6 Futhark arrays are REGULAR, not ragged — this is the one that matters for EXP-1

A jagged value ("row 0 has 3 elements, row 1 has 10") is **not an array in this language.** There is no first-class ragged type. The representation is:

```
flat values  +  per-row offsets  +  row count
```

This is confirmed in the language tour and manual, and it is the structural reason the
TICK kernel is written against a padded `[n][d]` adjacency. **Do not assume `d` can vary
per row.** The CSR layout is the honest representation, and it changes the kernel shape
entirely — that is not a detail, it is EXP-1's real content.

### 1.7 `^^` is BOOLEAN xor, not integer bitwise xor

Confusing the two produces a confusing error. There is **no verified language-level
integer XOR operator.** The `((a|b)-(a&b))` form used in `exp1` was found by execution,
not documented — use it because it is tested in the canary, not because it is endorsed.

### 1.8 `--memory-limit` is not out-of-core paging

It is a **runtime allocation limit**. It does not page to host, does not tile loops, and
does not split arrays. If an array exceeds device memory you get a runtime failure, not
a transparent fallback. Design for that.

### 1.9 Out-of-bounds and wraparound have no portable clean diagnostic

Per the manual: invalid indexing, integer wraparound, and invalid race assumptions can
produce **wrong results or crashes** with no diagnostic. This is directly the EXP-7
question, and the documentation says the answer is *yes, it can go quiet*. Budget for
independent verification of every number, not just the array sizes.

### 1.10 Failure to fuse is not proof of impossibility

Fusion is index- and shape-dependent, and the compiler may split two fusible operations
to reduce register pressure or launch count. **A disappointing kernel count is a
performance result, not a language verdict.** Use the inspection workflow before
concluding anything:

```bash
futhark dev --kernel-history=history.dat program.fut
futhark inspect --kscript history.dat program.fut
```

Verify the exact argument order with `--help` on your patch — it is not a stable API.
And do not profile with instrumentation enabled; it changes the timing.

### 1.11 Annotations are load-bearing, not decoration

Omit the type annotation on a 3-D intermediate and you will annotate it as 2-D and get
`cannot match value of type`. Futhark's shape inference does not flow reliably through
`let`-bound multi-dimensional intermediates.

---

## PART 1.12 The result that matters most: padded compiles, CSR does not

Research on GPU graph-cell work gives the storage arithmetic directly:

```
padded work = O(V * d_max)        CSR work = O(V + L)
d_max / mean_degree = wasted-slot ratio
```

| graph | V | d_max/mean | padded 32-bit IDs | CSR indices+offsets |
|---|---:|---:|---:|---:|
| ca-GrQc | 34,761 | 15.4x | 45.2 MB | 3.08 MB |
| ca-HepPh | 27,770 | 11.3x | 31.8 MB | 2.93 MB |
| wiki-Vote | 7,115,915 | 15.7x | 13.0 GB | 858 MB |
| soc-LiveJournal1 | 3,997,962 | **217.1x** | **60.2 GB** | 293 MB |

One hub sets the iteration width for the entire graph. That is the argument against padding
and it is arithmetic, not taste.

**So I wrote the CSR version. It does not compile.** `exp1c_tick_csr.fut` is checked in
unmodified as the failing control, and the failures are specific:

- the gather alone is fine — `nbrs[lo:hi]` is a dynamic-length slice, and
  `map (\j -> dials[idx[j]]) idx` compiles;
- combining that gather with elementwise arithmetic on the gathered rows does not:
  `Cannot apply "-" (invalid type). Expected: f64. Actual: f64.` — same printed type, and
  a let-bound 2-D intermediate produced by a map cannot be consumed by another map.

The padded kernel compiles *because* padding gives every row the same length, which gives
the type checker one shape variable that flows all the way through. Raggedness gives one
shape per row and Futhark will not carry it.

**The finding, stated plainly: the TICK bet is not blocked by the GPU. It is blocked by the
type system.** Padded is the wrong representation by up to 200x and it is the one that
compiles.

**Test next, in order:**
1. Flatten to 1-D by interleaving (all `m` dials of one neighbour contiguous), reduce in
   1-D where the type checker is happy, re-shape at the end.
2. If that fails, write the CSR kernel in CUDA C and use Futhark only for the padded
   comparison arm. That is a fine experimental design and still answers the question.
3. Either way, **report the crossing point of the padded kernel against degree skew.** A
   crossover at `d_max/mean ≈ 16` that moves to 217 is a strong localisation result.

## PART 2 — Verified programs, ready to run

### EXP-1 `exp1_tick.fut` — TICK as one kernel over the whole quilt  · COMPILES

**The architectural bet.** A cell's dial movement depends only on its graph neighbours, so
the whole graph should tick as one array operation rather than N method calls.

```
new_dials[i] = dials[i] + sum_j ( dials[adj[i][j]] - dials[i] )
```

This is `quilt-egg`'s resonance, vectorised. If the locality claim holds, this is large.
If it does not, you learn precisely where locality breaks — which is worth more.

**Run it against:** a per-cell Python loop on the same graph, and report the crossover
point. The interesting number is not "GPU is faster" but **at what n does it stop being
faster**, because for a sparse graph the adjacency has to be padded to a common degree
and the padding is pure waste.

**The correctness anchor:** a symmetric adjacency over a symmetric state with an *odd*
degree contributes zero, so TICK is the identity. If it drifts, the graph is malformed,
not slow. Check this before blaming the backend.

**Push past the edge:**
- **Ragged degree.** Real graphs are not regular. Change the input to a CSR-style flat
  adjacency plus an offsets array and measure. This is the load-bearing experiment.
- **d → 0.** What does Futhark do with a zero-width neighbour axis? Does it fold, crash,
  or silently produce garbage? A plausible-wrong-answer here is a finding.
- **Ordering.** Does permuting cell order change the result beyond floating-point noise?
  It should not, on a correct graph. If it does, you have found a real race.

### EXP-5 `exp5_witness_chain.fut` — the witness chain as a prefix problem  · COMPILES

The canon chain is currently `DECLARED-NOT-USED` — `prev_hash` exists but is zero in every
sampled live cell. This computes what the chain would be.

**The challenge to Futhark:** each cell's hash depends on the previous cell's hash. That is
a scan, not a map, and a sequential dependency is the classic place a data-parallel
language stops being data-parallel. The obvious sequential version needs a mutable
accumulator, which Futhark does not offer.

Three formulations are in the file:
- `chain_map` — N independent hashes. Trivially parallel. This is the lower bound on cost.
- `chain_prefix` — an O(n²) parallel prefix, written naively. Almost certainly slower.
- **You must write a real Blelloch-style parallel scan** and compare it against a
  sequential C baseline.

**The experiment is the comparison, not any single number.** If the scan beats the
sequential control at scale, Futhark's model survived a case designed to break it. If it
does not, that is the single most citable limitation in this report — and it argues the
witness chain should be *batched* (a Merkle root every K cells) rather than linear, which
is a design change to the canon, not just to the code.

---

## PART 3 — The experiment set, ordered by information per hour

### EXP-2 · Aperture: content-match vs motion-mask  · HIGH VALUE, WELL-POSED

A real fleet experiment, already characterised: a content matcher on a repeating texture
reports 33.2% of cells unmatched when **100%** of cells truly moved; a true baseline with
no motion reports 0.0%.

Port both to Futhark. This is a 2-D stencil, which is Futhark's home turf and the one
place the language should be unambiguously strong. **A negative result here is
information too**: if the GPU does not beat the CPU at this shape, nothing else will beat
it, and that bounds every other bet in this document.

### EXP-3 · Scatter  · EXPECTED TO BREAK

`scatter` is Futhark's documented weak spot. Drive it hard: 10⁶–10⁸ scattered writes with
colliding indices, and measure the write-amplification curve. Try to break it. The goal is
a *number* for the report, not a working kernel.

### EXP-4 · The canary at scale  · CHEAP CERTAINTY

FNV-1a 64 over the UTF-8 bytes of `café Δ 日本語` is `0x24a555471370b18d`. A port that
reproduces it lives at `../canary-3lang/canary.fut`. Run it over 10⁶ cells of varying body
length and measure throughput. Use it as the **calibration instrument** for every other
experiment: a number that has not been through the canary is not a number.

### EXP-6 · Negative space  · CREATIVE

The wheel's GAP cells were an artefact of a stop-word list filtering the corpus's own
vocabulary. Compute a *complement*: given N documents, the term-frequency space they do
**not** occupy. This is a histogram over an inverted index and it wants to be a reduction
over a very sparse matrix. If Futhark is bad at it, that is worth knowing before anyone
proposes it as a substrate.

### EXP-7 · Deliberately exceed device memory  · FIND THE CLIFF

Push the array past VRAM with `--memory-limit`. Where does it page, where does it thrash,
where does it fail cleanly, and where does it **silently produce wrong answers**? The last
one is the finding that matters. A language that degrades quietly is more expensive than
one that refuses.

### EXP-8 · The topology probe  · CONNECTED TO OPEN RESEARCH

A peer agent reported: *"embedding geometry is relation-blind in contract-decomposed
systems: it recovers identity and vocabulary, never topology or dialect role."*

There is a Futhark version of that question that is well-posed: take a graph, produce two
representations of it — one keyed by **identity** (cell IDs), one keyed by **local
neighbourhood statistics** — and ask whether a linear model on the second can recover the
first's *topology* (which cells are structurally equivalent, which are central). The
"dialect role" half of the claim maps onto: can you recover each cell's *function* in the
graph from local features alone, with no global view?

This is a real experiment, not a metaphor. Locality is the one thing Futhark gives you that
a Python graph library does not, so Futhark is the right place to ask whether local
features carry global structure.

---

## PART 4 — What to report back

For each experiment: the number, the crossover point against the CPU baseline, the backend
used, and **any place Futhark produced a plausible wrong answer instead of an error**.

That last item is the one that matters. Futhark has already produced one in this session —
a 2-D `+` that reads as `[][]f64` in both the expected and actual slot. A language that
fails quietly is a tool you have to instrument before you can trust, and the instrument
list is itself a deliverable.

## The canary rule for this whole lab

Any number that has not been through `0x24a555471370b18d` is not a measurement. Two
verification bugs this session were both in the *check*, not the code: a mistyped decimal
for a hex digest, and a `tr -d 'u64'` that deleted the `6` and `4` out of the number
itself. The check is where the errors live. Instrument it.
