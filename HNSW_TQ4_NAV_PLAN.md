# HNSW 4-bit Navigation Element: Design Sketch

A sketch of a 4-bit navigation element for pgvector's HNSW. It follows the GINGER paper's split (arXiv:2611.08306): a compact element tuple that beam search reads, and a separate re-rank tuple that's read only for the final candidates.

## 1. Opt-in and on-disk versioning

You'd enable it per index with a reloption, alongside `m` and `ef_construction` (`src/hnsw.c:97`):

```sql
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, quantizer = 'tq4', rerank = 'ntq12');   -- rerank: none | ntq10 | ntq12 | exact
```

Existing indexes keep working unchanged. Code that reads a page tells the formats apart by the tuple's `type` byte, and the metapage records which quantizer the index uses.

```c
/* hnsw.h */
#define HNSW_VERSION                 2
#define HNSW_ELEMENT_TUPLE_TYPE      1   /* existing: full-precision value */
#define HNSW_NEIGHBOR_TUPLE_TYPE     2   /* unchanged */
#define HNSW_NAV_ELEMENT_TUPLE_TYPE  3   /* new: 4-bit code */
#define HNSW_RERANK_TUPLE_TYPE       4   /* new: refinement bits or exact value */

/* page kinds stored in the currently-unused HnswPageOpaqueData.unused */
#define HNSW_PAGE_GRAPH   0              /* nav elements + neighbor tuples */
#define HNSW_PAGE_RERANK  1
```

New metapage fields go **after** the existing ones, so `HnswGetMetaPageInfo` (`src/hnswutils.c:312`) still reads version-1 metapages correctly. The `pd_lower` line at `src/hnswbuild.c:108` already covers whatever the struct's size is.

```c
typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	uint16		m;
	uint16		efConstruction;
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int16		entryLevel;
	BlockNumber insertPage;			/* graph-chain insert hint */
	/* --- version 2 --- */
	uint8		quantizer;			/* HNSW_QUANT_NONE / HNSW_QUANT_TQ4 */
	uint8		rerank;				/* NONE / NTQ10 / NTQ12 / EXACT */
	uint16		rotationR;			/* odd factor: dimensions = r · 2^l */
	uint32		rotationSeed;		/* regenerates signs + r×r orthogonal Q */
	BlockNumber rerankHeadBlkno;	/* first page of the re-rank chain */
	BlockNumber rerankInsertPage;
}			HnswMetaPageData;
```

The rotation is never stored. Each backend rebuilds it from `rotationSeed` and caches it in `rd_amcache`. The random signs and Walsh–Hadamard steps take no storage at all. The only real matrix is the r×r `Q`: 3×3 for 768, 1536 or 3072 dimensions. An awkward dimension like 1000 = 125·8 needs a 125×125 `Q`, about 62 KB, which is still cheap once cached per relation.

## 2. The navigation element tuple

```c
typedef struct HnswNavElementTupleData
{
	uint8		type;							/*  0  = HNSW_NAV_ELEMENT_TUPLE_TYPE */
	uint8		level;							/*  1 */
	uint8		deleted;						/*  2 */
	uint8		version;						/*  3 */
	ItemPointerData heaptids[HNSW_HEAPTIDS];	/*  4  60 B, unchanged */
	ItemPointerData neighbortid;				/* 64 */
	ItemPointerData reranktid;					/* 70  new: → HnswRerankTuple */
	float4		scale;							/* 76  per-vector 4-bit scale (paper's s_4) */
	float4		norm;							/* 80  ‖x‖, needed for L2 */
	uint8		code[FLEXIBLE_ARRAY_MEMBER];	/* 84  ceil(dim/2) B, 2 coords/byte */
}			HnswNavElementTupleData;

#define HNSW_NAV_ELEMENT_TUPLE_SIZE(dim) \
	MAXALIGN(offsetof(HnswNavElementTupleData, code) + ((dim) + 1) / 2)
```

The first 70 bytes are identical to `HnswElementTupleData` (`src/hnsw.h:376`). So vacuum, `HnswFreeOffset` and `AddDuplicateOnDisk` can keep reading `deleted`, `version`, `heaptids` and `neighbortid` through a shared prefix, with no change. The varlena header is dropped because the code length is fixed by `metap->dimensions`.

The neighbor tuple doesn't change at all. It still contains `(blk, off)` pointers to element tuples. Those pointers just land on smaller tuples now.

## 3. The re-rank tuple, on its own page chain

```c
typedef struct HnswRerankTupleData
{
	uint8		type;							/* 0 = HNSW_RERANK_TUPLE_TYPE */
	uint8		version;						/* 1 must match the element's version */
	uint16		unused;
	float4		scale;							/* 4 refined scale s_{4+k} (unused for EXACT) */
	uint8		data[FLEXIBLE_ARRAY_MEMBER];	/* 8 k bits/coord packed, or a full varlena */
}			HnswRerankTupleData;
```

Re-rank tuples live on **separate pages** (`HNSW_PAGE_RERANK`), in a second chain that starts at `metap->rerankHeadBlkno`. If they shared pages with the graph, the density gain would be lost. Beam search never touches these pages; only the final re-rank does.

```
graph chain (dense, hot):
  blk 1  [nav A][nbrs A][nav B][nbrs B][nav C][nbrs C] … ×7 → nextblkno
  blk 2  …
re-rank chain (cold, read ~ef_search tuples per query):
  blk K  [rr A][rr B][rr C][rr D][rr E] → nextblkno
```

Every tuple in a given index now has a **fixed size**: nav tuples per level, re-rank tuples exactly. A deleted slot can therefore always be reused by a new element of the same level or lower. That makes the size checks in `HnswFreeOffset` trivial, and the FSM path used today for `sparsevec` isn't needed.

## 4. Size per page

These assume 8 KB pages, `m = 16`, and a level-0 element. A level-0 neighbor tuple is 200 bytes, and every tuple also costs a 4-byte line pointer.

| Dims | Today: element + neighbors | Per page | Nav: element + neighbors | Per page | Re-rank `ntq12` | Per page |
|---|---|---|---|---|---|---|
| 768 | 3152 + 200 | 2 | 472 + 200 | **12** | 776 | 10 |
| 1536 | 6224 + 200 | 1 | 856 + 200 | **7** | 1544 | 5 |
| 3072 | not supported (`HNSW_MAX_DIM` 2000) | – | 1624 + 200 | **4** | 3080 | 2 |

For 1536 dimensions, per element:
- **Today:** one 8 KB page.
- **Nav + `ntq12`:** about 1.2 KB of graph pages plus 1.6 KB of re-rank pages, roughly 2.8 KB. That's about 46% of the raw float size, in line with the paper's ~43%.
- **Graph pages only** (the working set every query walks): **7× smaller.** For 10M × 1536 dimensions, that's about 12 GB of graph pages versus about 82 GB today.
- **Nav + `exact`:** about 15% *larger* than today overall. The 7× smaller graph working set still applies, and results keep exact ordering.

A side benefit: the 4-bit code makes the 2000-dimension limit go away for the graph pages. With `rerank = exact`, though, a 3072-dimension float vector (about 12 KB) wouldn't fit on a re-rank page. That combination would need either `ntq*` re-ranking or re-ranking from the table row.

## 5. Distance computation

At the start of a scan, rotate the query once and keep it alongside `q->value`:

```c
typedef struct HnswPreparedQuery
{
	float	   *qr;			/* R·q, length = dimensions (padded) */
	float		qnorm2;		/* ‖q‖² */
}			HnswPreparedQuery;
```

Reference scalar kernel. A SIMD version can come later, using the same dispatch pattern as `halfutils.c` and `bitutils.c`:

```c
static const float tq4_centroids[16] = { /* Lloyd–Max for N(0,1), sign + 3-bit magnitude */ };

static float
Tq4Dot(const float *qr, const uint8 *code, int dim)
{
	float		acc = 0;

	for (int i = 0; i < dim / 2; i++)
	{
		uint8		b = code[i];

		acc += qr[2 * i] * tq4_centroids[b & 0x0F];
		acc += qr[2 * i + 1] * tq4_centroids[b >> 4];
	}
	return acc;
}

/* est ⟨q,x⟩ = etup->scale * Tq4Dot(...)
 * L2²  ≈ qnorm2 + norm² − 2·est      (vector_l2_ops)
 * -IP  ≈ −est                        (vector_ip_ops; cosine is already normalized
 *                                     by HnswFormIndexValue, so norm = 1) */
```

This takes the place of `HnswGetDistance(q->value, PointerGetDatum(&etup->data), support)` at `src/hnswutils.c:572`. The cleanest approach is a small quantizer vtable hung off `HnswSupport`, so the rest of `HnswSearchLayer` doesn't change:

```c
typedef struct HnswQuantizer
{
	Size		(*navSize) (int dim);
	Size		(*rerankSize) (int dim);
	void		(*encode) (Datum value, const HnswRotation *, HnswNavElementTuple, HnswRerankTuple);
	void		(*prepare) (Datum query, const HnswRotation *, HnswPreparedQuery *);
	float		(*navDistance) (const HnswPreparedQuery *, HnswNavElementTuple);
	float		(*rerankDistance) (const HnswPreparedQuery *, HnswRerankTuple);
	void		(*decode) (HnswNavElementTuple, float *out);	/* for element-vs-element */
}			HnswQuantizer;
```

## 6. Code paths that change

| File | Change |
|---|---|
| `src/hnsw.h` | The new structs and size macros, metapage v2, page kinds in `opaque.unused`, a `rerankPage`/`rerankOffno` pair on `HnswElementData` |
| `src/hnsw.c` | `quantizer` and `rerank` reloptions, added to `hnswoptions()` at `:255` |
| `src/hnswbuild.c` | The in-memory build **keeps exact vectors** for best graph quality. `CreateGraphPages` (`:147`) writes nav + neighbor tuples to the graph chain and appends re-rank tuples to a second chain, setting `reranktid`. `WriteNeighborTuples` is unchanged. |
| `src/hnswutils.c` | `HnswLoadElementImpl` (`:550`) switches on the tuple type. `CheckElementCloser` (`:1237`) compares two *stored* elements, so it decodes one code to floats, or uses an asymmetric query-vs-code distance. |
| `src/hnswinsert.c` | `AddElementOnDisk` (`:149`) also places a re-rank tuple, at `rerankInsertPage` or in a recycled slot. `FindDuplicateOnDisk` (`:706`) compares bytes with `datumIsEqual`; on codes, that would **merge distinct vectors that happen to share a code**. So dedupe only when `rerank = exact` and the exact bytes match, or require both the code and the refinement bits to match. |
| `src/hnswscan.c` | After the layer-0 `HnswSearchLayer` in `GetScanItems`, **re-rank `w`**: read each candidate's re-rank tuple (checking `version`), recompute the distance, and re-sort. Iterative scans re-rank each resumed batch the same way. |
| `src/hnswvacuum.c` | `MarkDeleted` (`:731`) also zeroes the re-rank tuple and bumps its version. `RepairGraph` uses decoded distances. |

**Why re-rank inside the access method instead of setting `xs_recheckorderby = true` and letting Postgres reorder:** Postgres's reorder queue assumes the index's distance is a *lower bound* on the exact distance. Quantized estimates can be on either side. When the exact distance comes out smaller, the executor raises "index returned tuples in wrong order." Re-ranking ourselves keeps today's contract (`xs_recheckorderby = false`, approximate order, `src/hnswscan.c:332`).

## 7. Open risks to measure

1. **Insert-time graph quality.** The build uses exact distances, but later inserts and vacuum repairs prune neighbor lists using codes. Recall could drift on tables with heavy churn.
2. **Page locality.** Packing 7 nodes per page shrinks the working set, but nodes are placed in insertion order, so a hop rarely lands on a page it just read. The analogue of the paper's graph reordering is cheap here: `CreateGraphPages` could write elements in BFS or partition order. That's a follow-up, but it's where same-page hits would actually come from.
3. **Re-rank I/O.** A query does `ef_search` (40 by default) random re-rank reads. That's small next to the hundreds of graph nodes visited, but for `rerank = exact` each of those is a full page.
4. **`heaptids[10]` costs 60 bytes,** about 7% of a 1536-dimension nav tuple. A later version could shrink it to a few inline slots plus an overflow.

## Suggested first prototype

A sensible first prototype would be the header changes, the encoder, and the scalar `Tq4Dot`, wired in behind `quantizer = 'tq4'` with `rerank = exact` only. That isolates the navigation change from refinement-code accuracy, so a recall/QPS comparison against today's index shows exactly what the 4-bit graph costs and gains.

## Implementation notes (branch `hnsw-tq4-nav`)

Where the implementation differs from the sketch above:

- **No new element tuple type.** Quantized elements keep `type = HNSW_ELEMENT_TUPLE_TYPE`. The `data` field holds an `HnswTqCode` varlena instead of a `Vector`, marked by `-1` in the slot where `Vector.dim` would be. All page-walking code (vacuum, `HnswFreeOffset`, duplicate handling) works unchanged. Cost: the element header is 96 bytes instead of 84.
- **`reranktid` lives in the `HnswTqCode` header**, not in the element tuple.
- **Re-rank tuples (`type = 3`)** sit on their own page chain (`metap->rerankHeadBlkno` / `rerankInsertPage`). They are allocated before the element, without holding graph locks. Their slots are reused on their own, not paired with element slots. Vacuum tombstones them in the same WAL record as the element.
- **Rotation:** the r×r matrix `Q` is stored in the metapage, and the signs are regenerated from the stored seed. Padding uses an odd `r ≤ 31` (768/1536/3072 → r = 3, no padding).
- **In-memory values:** queries and new elements become an `HnswTqValue` holding `R·x` (marker `-2`). Loaded elements decode to `s_4·ỹ`. The in-memory build phase uses exact distances. On-disk inserts and vacuum select neighbors with the 4-bit estimates.
- **Opclasses:** L2, inner product and cosine (cosine reuses the inner-product path on normalized vectors).
- **Codebooks** are generated by `scripts/gen_hnsw_tq_tables.py` into `src/hnswtq_tables.h` (D₄ = 0.0095, D₁₀ = 2.7e-6, D₁₂ = 1.7e-7).

Known limitations: iterative scans raise an error. No duplicate merging for on-disk inserts into quantized indexes. The 2000-dimension cap is unchanged. Kernels are scalar. If an insert fails after its re-rank tuple is added, that tuple is orphaned. Re-rank reads re-check the element's version but not the re-rank tuple's owner.
