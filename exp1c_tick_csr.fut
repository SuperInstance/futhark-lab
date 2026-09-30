-- EXP-1c: TICK in CSR form -- THE EXPERIMENT THAT DID NOT COMPILE, and why it matters.
--
-- exp1_tick.fut uses a PADDED adjacency [n][d]i64. The arithmetic says that is the wrong
-- shape: padded work is O(V * d_max), CSR work is O(V + L), and d_max/mean_degree is the
-- wasted-slot ratio. Real SNAP graphs:
--
--     ca-GrQc              34,761 nodes     15.4x    padded  45.2 MB  vs  CSR   3.08 MB
--     ca-HepPh             27,770 nodes     11.3x    padded  31.8 MB  vs  CSR   2.93 MB
--     wiki-Vote         7,115,915 nodes     15.7x    padded  13.0 GB  vs  CSR  858   MB
--     soc-LiveJournal1  3,997,962 nodes    217.1x    padded  60.2 GB  vs  CSR  293   MB
--
-- One hub sets the iteration width for the whole graph. That is the argument, and it is
-- arithmetic rather than taste. So the honest next step is CSR:
--
--     dials   : [n][m]f64    each cell's own state
--     offsets : [n+1]i64     cell i's neighbours are nbrs[offsets[i] : offsets[i+1]]
--     nbrs    : [L]i64        flat neighbour indices
--
-- ── WHAT BLOCKED IT, precisely ───────────────────────────────────────────────────────
--
-- The gather itself is fine. Dynamic-bound slicing works:
--     let idx = nbrs[lo:hi]                      -- runtime-length slice
--     let gathered = map (\j -> dials[idx[j]]) idx   -- compiles
--
-- What does NOT compile is combining that with elementwise arithmetic on the gathered
-- rows. Every form below was tried and every one fails:
--
--   map (\j -> map (\q -> dials[idx[j]][q] - self[q]) self) idx
--       -> Cannot apply "-" (invalid type). Expected: f64. Actual: f64.
--          The two types print identically. The shape variable does not unify.
--
--   let gathered = ... in let dif = map2 (\ra rb -> map2 (-) ra rb) gathered selfs
--       -> Cannot apply "map" to "gathered" (invalid type). Expected [][]f64. Actual [][]f64.
--          A let-bound 2-D intermediate produced by a map cannot be consumed by another map.
--
--   let dif = ... in reduce (+) 0.0f64 (map (\j -> reduce (+) 0.0f64 dif[j]) gathered)
--       -> same class of failure.
--
-- The padded kernel in exp1_tick.fut compiles because padding makes every row the SAME
-- length, which gives the type checker a single shape variable that flows all the way
-- through. The ragged version has one shape per row, and Futhark will not carry it.
--
-- ── WHY THIS IS THE MOST USEFUL RESULT IN THE LAB ────────────────────────────────────
--
-- The padded version compiles and is the wrong representation by 1-2 orders of magnitude.
-- The CSR version is the right representation and does not typecheck. That is a real
-- property of the language, and it is the property that matters for TICK-as-one-kernel:
-- the bet is not blocked by the GPU, it is blocked by the type system.
--
-- WORKAROUND TO TEST: flatten to 1-D by interleaving (cell-major, all m dials of one
-- neighbour contiguous), reduce in 1-D where the type checker is happy, then re-shape.
-- If that also fails, the honest conclusion is that Futhark wants a padded dense operator
-- and the CSR path belongs in CUDA C -- which is a fine answer, and still an answer.
--
-- This file is kept compiling-or-not on purpose. It is the experiment.

def tick_csr (dials: [][]f64) (offsets: []i64) (nbrs: []i64) : [][]f64 =
  let nc = reduce i64.max 0 offsets - 1
  in map (\i ->
        let lo = offsets[i]
        let hi = offsets[i+1]
        let self = dials[i]
        let idx = nbrs[lo:hi]
        -- the gather alone, which does compile:
        let gathered = map (\j -> dials[idx[j]]) idx
        -- and the arithmetic that does not. Left in place as the failing control.
        let dif = map2 (\ra rb -> map2 (-) ra rb) gathered (replicate (reduce i64.max 0 idx) self)
        in reduce (+) 0.0f64 (map (\j -> reduce (+) 0.0f64 dif[j]) gathered))
     (iota nc)

def main [n][m] (dials: [n][m]f64) (offsets: [n+1]i64) (nbrs: [l]i64) : [n][m]f64 =
  tick_csr dials offsets nbrs
