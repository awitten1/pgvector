#include "postgres.h"

#include <math.h>

#include "fmgr.h"
#include "halfutils.h"
#include "halfvec.h"
#include "hnsw.h"
#include "rabitq.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "vector.h"

/*
 * Experimental: estimate distances during HNSW scans with RaBitQ.
 *
 * Indexes created with the rabitq_bits option store a code for each element.
 * Scans estimate distances from the stored codes, and the executor reranks
 * with exact distances from the heap (see hnswgettuple).
 *
 * By default, the code is stored instead of the value (as a varlena with the
 * factors followed by the code), so the index does not contain the original
 * vectors. In-memory builds still use exact distances, but inserts, vacuum,
 * and the on-disk phase of builds only have codes for existing elements, so
 * they use estimated distances: asymmetric ones (value vs code) to search
 * for the neighbors of a new element, and symmetric ones (code vs code)
 * between existing elements.
 *
 * With rabitq_keep_vectors, the code is stored after the value instead, and
 * builds, inserts, and vacuum use exact distances.
 *
 * Scans of indexes with vectors can also quantize values on the fly, which
 * is slow and only useful for measuring recall.
 */

/* Fixed so codes and queries always use the same rotation */
#define HNSW_RABITQ_SEED 42

/* FHT-Kac rotation requirement */
#define HNSW_RABITQ_MIN_DIM 64

#define HNSW_RABITQ_FACTORS_SIZE (RABITQ_NUM_FACTORS * sizeof(float))

struct HnswRabitq
{
	RabitqContext *ctx;
	bool		isHalf;
	bool		stored;			/* use codes stored in the index */
	bool		codesOnly;		/* codes are stored instead of values */
	bool		hasQuery;
	int			dim;
	Size		codeBytes;		/* excluding factors */
	float	   *buf;			/* for converting halfvec to float */
};

/*
 * Free the C++ context when the memory context is reset or deleted,
 * including on error
 */
static void
HnswRabitqFree(void *arg)
{
	HnswRabitq *rabitq = (HnswRabitq *) arg;

	rabitq_free(rabitq->ctx);
	rabitq->ctx = NULL;
}

static void
ReportError(const char *action, const char *errbuf)
{
	ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("RaBitQ %s failed: %s", action, errbuf)));
}

/*
 * Get the type and metric from the distance function
 */
static bool
GetTypeAndMetric(HnswSupport * support, bool *isHalf, RabitqMetric * metric)
{
	char	   *name = get_func_name(support->procinfo->fn_oid);

	if (name == NULL)
		return false;

	/* Cosine uses negative inner product on normalized values */
	if (strcmp(name, "vector_l2_squared_distance") == 0)
	{
		*isHalf = false;
		*metric = RABITQ_METRIC_L2;
	}
	else if (strcmp(name, "vector_negative_inner_product") == 0)
	{
		*isHalf = false;
		*metric = RABITQ_METRIC_IP;
	}
	else if (strcmp(name, "halfvec_l2_squared_distance") == 0)
	{
		*isHalf = true;
		*metric = RABITQ_METRIC_L2;
	}
	else if (strcmp(name, "halfvec_negative_inner_product") == 0)
	{
		*isHalf = true;
		*metric = RABITQ_METRIC_IP;
	}
	else
		return false;

	return true;
}

/*
 * Get a value as floats
 */
static const float *
GetFloats(HnswRabitq * rabitq, Pointer value)
{
	if (rabitq->isHalf)
	{
		HalfVector *vec = (HalfVector *) value;

		if (vec->dim != rabitq->dim)
			elog(ERROR, "unexpected dimensions for RaBitQ");

		for (int i = 0; i < vec->dim; i++)
			rabitq->buf[i] = HalfToFloat4(vec->x[i]);

		return rabitq->buf;
	}
	else
	{
		Vector	   *vec = (Vector *) value;

		if (vec->dim != rabitq->dim)
			elog(ERROR, "unexpected dimensions for RaBitQ");

		return vec->x;
	}
}

/*
 * Set up RaBitQ for an index
 *
 * Allocates in CurrentMemoryContext, and the result is freed when that
 * context is reset or deleted.
 */
HnswRabitq *
HnswRabitqInit(HnswSupport * support, int dimensions, int bits, bool codesOnly, Relation index)
{
	HnswRabitq *rabitq;
	MemoryContextCallback *callback;
	bool		isHalf;
	RabitqMetric metric;
	char		errbuf[256];

	if (!GetTypeAndMetric(support, &isHalf, &metric))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("RaBitQ only supports vector and halfvec with L2, inner product, or cosine distance")));

	if (dimensions < HNSW_RABITQ_MIN_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("RaBitQ requires at least %d dimensions", HNSW_RABITQ_MIN_DIM)));

	rabitq = palloc0_object(HnswRabitq);
	rabitq->isHalf = isHalf;
	rabitq->codesOnly = codesOnly;
	rabitq->dim = dimensions;
	if (isHalf)
		rabitq->buf = palloc(mul_size(sizeof(float), dimensions));

	/* Register cleanup before creating so the context can never leak */
	callback = palloc_object(MemoryContextCallback);
	callback->func = HnswRabitqFree;
	callback->arg = rabitq;
	MemoryContextRegisterResetCallback(CurrentMemoryContext, callback);

	if (rabitq_create(dimensions, bits, metric, HNSW_RABITQ_SEED, &rabitq->ctx, errbuf, sizeof(errbuf)) != 0)
		ReportError("setup", errbuf);

	rabitq->codeBytes = rabitq_code_bytes(rabitq->ctx);

	/* Use the centroid from the index */
	if (index != NULL)
	{
		float	   *centroid = HnswLoadCentroid(index, dimensions);

		if (centroid != NULL)
		{
			HnswRabitqSetCentroid(rabitq, centroid);
			pfree(centroid);
		}
	}

	return rabitq;
}

/*
 * Set the centroid that values are encoded relative to
 *
 * Must be called before encoding or setting the query
 */
void
HnswRabitqSetCentroid(HnswRabitq * rabitq, const float *centroid)
{
	char		errbuf[256];

	if (rabitq_set_centroid(rabitq->ctx, centroid, errbuf, sizeof(errbuf)) != 0)
		ReportError("centroid setup", errbuf);

	rabitq->hasQuery = false;
}

/*
 * Add a value to sums (for calculating the centroid)
 */
void
HnswRabitqAddToSums(HnswRabitq * rabitq, Pointer value, double *sums)
{
	const float *x = GetFloats(rabitq, value);

	for (int i = 0; i < rabitq->dim; i++)
		sums[i] += x[i];
}

/*
 * Encode a value
 */
static void
Encode(HnswRabitq * rabitq, Pointer value, char *dst)
{
	float		factors[RABITQ_NUM_FACTORS];
	char		errbuf[256];

	if (rabitq_encode(rabitq->ctx, GetFloats(rabitq, value), (uint8_t *) dst + HNSW_RABITQ_FACTORS_SIZE, factors, errbuf, sizeof(errbuf)) != 0)
		ReportError("encoding", errbuf);

	memcpy(dst, factors, HNSW_RABITQ_FACTORS_SIZE);
}

/*
 * Get the size of the factors and code stored for each element
 */
Size
HnswRabitqCodeSize(HnswRabitq * rabitq)
{
	return add_size(HNSW_RABITQ_FACTORS_SIZE, rabitq->codeBytes);
}

/*
 * Get the size of an element tuple (rabitq can be NULL)
 */
Size
HnswRabitqElementTupleSize(HnswRabitq * rabitq, Size valueSize)
{
	/* The value is the code if codes only */
	if (rabitq == NULL || rabitq->codesOnly)
		return HNSW_ELEMENT_TUPLE_SIZE(valueSize);

	return HNSW_ELEMENT_TUPLE_CODE_SIZE(valueSize, HnswRabitqCodeSize(rabitq));
}

/*
 * Get the max size of an element tuple
 */
Size
HnswRabitqMaxElementTupleSize(HnswRabitq * rabitq)
{
	Size		valueSize;

	if (rabitq->codesOnly)
		valueSize = add_size(VARHDRSZ, HnswRabitqCodeSize(rabitq));
	else
		valueSize = rabitq->isHalf ? HALFVEC_SIZE(rabitq->dim) : VECTOR_SIZE(rabitq->dim);

	return HnswRabitqElementTupleSize(rabitq, valueSize);
}

/*
 * Get the location of the factors and code in an element tuple
 */
static char *
GetStoredCode(HnswElementTuple etup)
{
	return ((char *) etup) + HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(&etup->data));
}

/*
 * Add the code for the value to an element tuple (rabitq can be NULL)
 *
 * The value must be set, and the tuple must be allocated with
 * HnswRabitqElementTupleSize.
 */
void
HnswRabitqSetElementTuple(HnswRabitq * rabitq, HnswElementTuple etup)
{
	/* The value is the code if codes only */
	if (rabitq == NULL || rabitq->codesOnly)
		return;

	Encode(rabitq, (Pointer) &etup->data, GetStoredCode(etup));
}

/*
 * Form the code for a value as a varlena, which is stored instead of the
 * value if codes only
 */
Pointer
HnswRabitqFormCode(HnswRabitq * rabitq, Pointer value)
{
	Size		size = add_size(VARHDRSZ, HnswRabitqCodeSize(rabitq));
	Pointer		result = palloc0(size);

	SET_VARSIZE(result, size);
	Encode(rabitq, value, VARDATA(result));

	return result;
}

/*
 * Get the factors and code from a varlena
 */
static char *
GetCode(HnswRabitq * rabitq, Pointer value)
{
	/* Safety check */
	if (VARSIZE_ANY_EXHDR(value) != HnswRabitqCodeSize(rabitq))
		elog(ERROR, "hnsw element does not have a RaBitQ code");

	return VARDATA_ANY(value);
}

/*
 * Set the query from a value
 */
void
HnswRabitqSetQuery(HnswRabitq * rabitq, Pointer value)
{
	char		errbuf[256];

	if (rabitq_set_query(rabitq->ctx, GetFloats(rabitq, value), errbuf, sizeof(errbuf)) != 0)
		ReportError("query preparation", errbuf);

	rabitq->hasQuery = true;
}

bool
HnswRabitqHasQuery(HnswRabitq * rabitq)
{
	return rabitq->hasQuery;
}

/*
 * Estimate the distance between two codes
 */
double
HnswRabitqSymmetricDistance(HnswRabitq * rabitq, Pointer a, Pointer b)
{
	char	   *srcA = GetCode(rabitq, a);
	char	   *srcB = GetCode(rabitq, b);
	float		factorsA[RABITQ_NUM_FACTORS];
	float		factorsB[RABITQ_NUM_FACTORS];
	float		distance;
	char		errbuf[256];

	memcpy(factorsA, srcA, HNSW_RABITQ_FACTORS_SIZE);
	memcpy(factorsB, srcB, HNSW_RABITQ_FACTORS_SIZE);

	if (rabitq_symmetric(rabitq->ctx, (const uint8_t *) srcA + HNSW_RABITQ_FACTORS_SIZE, factorsA,
						 (const uint8_t *) srcB + HNSW_RABITQ_FACTORS_SIZE, factorsB,
						 &distance, errbuf, sizeof(errbuf)) != 0)
		ReportError("estimation", errbuf);

	return distance;
}

/*
 * Prepare RaBitQ for a scan
 *
 * Returns NULL to use exact distances. indexBits is the bits per dimension
 * of the codes stored in the index (zero if none).
 */
HnswRabitq *
HnswRabitqBeginScan(Relation index, HnswSupport * support, Datum value, int dimensions, int indexBits, bool codesOnly)
{
	HnswRabitq *rabitq;
	bool		automatic = hnsw_rabitq_bits < 0;
	int			bits = automatic ? indexBits : hnsw_rabitq_bits;

	if (DatumGetPointer(value) == NULL)
		return NULL;

	if (codesOnly)
	{
		/* The index does not have values for other distances */
		if (bits != indexBits)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("index only supports hnsw.rabitq_bits = %d", indexBits),
					 errhint("The index stores RaBitQ codes without vectors.")));

		if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("iterative scans are not supported for hnsw indexes with RaBitQ codes without vectors"),
					 errhint("Set hnsw.iterative_scan = off.")));

		rabitq = HnswRabitqInit(support, dimensions, bits, true, index);
		rabitq->stored = true;
		HnswRabitqSetQuery(rabitq, (Pointer) PG_DETOAST_DATUM(value));

		return rabitq;
	}

	if (bits == 0)
		return NULL;

	if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
	{
		/* Use exact distances unless RaBitQ was explicitly requested */
		if (automatic)
			return NULL;

		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hnsw.rabitq_bits does not support iterative scans"),
				 errhint("Set hnsw.iterative_scan = off.")));
	}

	/* The centroid is not needed to quantize on the fly, but is used if present */
	rabitq = HnswRabitqInit(support, dimensions, bits, false, index);

	/* Quantize on the fly if the index does not have matching codes */
	rabitq->stored = bits == indexBits;

	HnswRabitqSetQuery(rabitq, (Pointer) PG_DETOAST_DATUM(value));

	return rabitq;
}

/*
 * Estimate the distance between the scan query and an element
 *
 * The estimate has the same ordering as the support function (squared L2,
 * or 1 - inner product instead of negative inner product), but not the same
 * values, so it must only be compared with other estimates.
 */
double
HnswRabitqDistance(HnswRabitq * rabitq, HnswElementTuple etup, Size etupSize)
{
	float		distance;
	char		errbuf[256];

	if (rabitq->codesOnly)
	{
		char	   *src = GetCode(rabitq, (Pointer) &etup->data);
		float		factors[RABITQ_NUM_FACTORS];

		memcpy(factors, src, HNSW_RABITQ_FACTORS_SIZE);

		if (rabitq_compute(rabitq->ctx, (const uint8_t *) src + HNSW_RABITQ_FACTORS_SIZE, factors, &distance, errbuf, sizeof(errbuf)) != 0)
			ReportError("estimation", errbuf);
	}
	else if (rabitq->stored)
	{
		Size		valueSize = VARSIZE_ANY(&etup->data);
		char	   *src = GetStoredCode(etup);
		float		factors[RABITQ_NUM_FACTORS];

		/* Safety check */
		if (HnswRabitqElementTupleSize(rabitq, valueSize) > etupSize)
			elog(ERROR, "hnsw element does not have a RaBitQ code");

		memcpy(factors, src, HNSW_RABITQ_FACTORS_SIZE);

		if (rabitq_compute(rabitq->ctx, (const uint8_t *) src + HNSW_RABITQ_FACTORS_SIZE, factors, &distance, errbuf, sizeof(errbuf)) != 0)
			ReportError("estimation", errbuf);
	}
	else
	{
		if (rabitq_estimate(rabitq->ctx, GetFloats(rabitq, (Pointer) &etup->data), &distance, errbuf, sizeof(errbuf)) != 0)
			ReportError("estimation", errbuf);
	}

	return distance;
}
