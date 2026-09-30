-- EXP-5: the witness chain, FNV-1a 64 per cell, one pass over the whole graph.
--
-- The canon chain is DECLARED-NOT-USED: prev_hash exists but is zero in every sampled
-- live cell. The cheap experiment is to compute what the chain *would* be, for every
-- cell, and see whether chaining is even the right shape.
--
-- CHALLENGE IT PUTS TO FUTHARK: this is a SCAN. Each cell's hash depends on the previous
-- cell's hash, so the parallelism is a prefix-sum (scan) over a sequential dependency,
-- not an independent map. That is the classic case where a data-parallel language stops
-- being data-parallel. Implement the naive map; then try a parallel-prefix version and
-- measure whether the extra complexity buys anything. That comparison IS the experiment.

def xor64 (a: u64) (b: u64) : u64 = (a | b) - (a & b)

def fnv1a64 (s: []u64) : u64 =   -- unsized param: [k] size decls are entry-point-only   -- byte values carried as u64: Futhark does no promotion
  let step (h: u64) (b: u64) : u64 = xor64 h b * 0x100000001b3
  in foldl step 0xcbf29ce484222325 s

-- One hash per cell, independent. This is the version that parallelises.
def chain_map [n][k] (bodies: [n][k]u64) : [n]u64 = map fnv1a64 bodies

-- Parallel-prefix attempt: compute the independent hashes first (parallel), then a
-- Blelloch-style scan to fold them into a chain. If the scan costs more than the naive
-- sequential loop, Futhark's data-parallel model has met a real wall and that is the
-- finding to report.
def chain_prefix [n][k] (bodies: [n][k]u64) : [n]u64 =
  let base = chain_map bodies
  let acc (i: i64) : u64 =
    let go (a: u64) (j: i64) : u64 = xor64 base[j] a
    in foldl go 0x0000000000000000u64 (iota (i + 1))
  in map acc (iota n)

def main [n][k] (bodies: [n][k]u64) : [n]u64 = chain_prefix bodies
