-- EXP-1: TICK as ONE kernel over the whole quilt.
--
-- THE BET. A cell's dial movement depends only on its graph neighbours. If that is true,
-- the entire graph ticks as one array operation instead of N method calls. Either the
-- fleet's central locality claim is true and this is enormous, or it is false and we learn
-- exactly where locality breaks. Both outcomes beat the port.
--
--     new_dials[i] = dials[i] + sum_j ( dials[adj[i][j]] - dials[i] )
--
-- This is quilt-egg's resonance, vectorised.
--
-- ── FUTHARK EDGES FOUND BY EXECUTION, NOT BY READING ────────────────────────────
--  1. `map` and `map2` on a 2-D array are ROW-WISE, not elementwise. `map (+1.0)` on an
--     [n][d]f64 is a type error, because it wants to add 1.0 to a whole row. Elementwise
--     on 2-D means a nested map. This is the single most likely thing to waste an hour on.
--  2. A gather and an arithmetic op cannot share a lambda:
--         map (\j -> dials[adj[i][j]] - dials[i])     -- type error, shape won't infer
--     The gather alone is fine. Split into passes and it compiles.
--  3. Size parameters `[n][d]` are only valid on an ENTRY POINT. On an internal `def`
--     they are "Unknown name".
--  4. `type`, `entry`, and `entrypoint` are all reserved words.
--  5. There is no integer XOR; it is (a|b)-(a&b). No implicit numeric promotion.
--  6. The reduction axis must be reduced in a SEPARATE pass with its own type
--     annotation, or you annotate the 3-D intermediate as 2-D and it does not unify.
--
-- ── WHERE THIS PUSHES PAST THE EDGE ────────────────────────────────────────────
--  * The neighbour axis `d` is padded to a common degree so the reduction is dense.
--    A real graph is ragged. exp1b_ragged.fut restores the real degree distribution so
--    the cost of faking regularity is measured rather than hidden.
--  * adj is a dense [n][d]i64. A sparse graph is [n][d] in the limit of many padding
--    columns. The edge is where d must become ragged or the adjacency must become CSR,
--    and Futhark's `scatter` is the suspected cliff.
--  * CORRECTNESS ANCHOR: a symmetric adjacency over a symmetric state with an odd
--    degree gives zero contribution, so TICK is the identity. If that drifts, the graph
--    is malformed rather than slow -- check it before blaming the backend.

def tick [n][d][m] (adj: [n][d]i64) (dials: [n][m]f64) : [n][m]f64 =
  -- Accumulate the neighbour contribution with a fold over d. A 3-D intermediate
  -- (nb - sb) does not unify in the type checker; a row-wise map2 per neighbour
  -- slot does. That is a compiler limitation, not a math one.
  -- (pass 3 kept for reference; the live path accumulates over d instead)
  -- pass 4: accumulate over the neighbour axis with a fold, which sidesteps the
  -- 3-D reduction that does not unify. One full [n][m] pass per neighbour slot.
  let contrib : [n][m]f64 =
    foldl (\(acc: [n][m]f64) (j: i64) ->
             let step : [n][m]f64 = map (\i -> map (\q -> dials[adj[i][j]][q] - dials[i][q]) (iota m)) (iota n)
             in map2 (\ra rb -> map2 (-) ra rb) acc step)
           (replicate n (replicate m 0.0f64))
           (iota d)
  -- the final add is 2-D, so it also goes row-wise. Nothing in Futhark adds two
  -- 2-D arrays directly; that is a type error, not a performance question.
  in map2 (\rd rc -> map2 (+) rd rc) dials contrib

def main [n][d][m] (adj: [n][d]i64) (dials: [n][m]f64) : [n][m]f64 = tick adj dials
