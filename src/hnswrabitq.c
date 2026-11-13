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
 * This is stage 1a of the experiment: the index format is unchanged, and each
 * element's vector is quantized on the fly when it is loaded. That is slower
 * than exact distances, but it shows how recall depends on the rerank depth
 * (ef_search) without changing anything on disk. Reranking is done by the
 * executor (see hnswgettuple).
 */

/* Fixed so that results are reproducible */
#define HNSW_RABITQ_SEED 42

/* FHT-Kac rotation requirement */
#define HNSW_RABITQ_MIN_DIM 64

struct HnswRabitq
{
	RabitqContext *ctx;
	bool		isHalf;
	int			dim;
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
 * Prepare RaBitQ for a scan
 *
 * Returns NULL if RaBitQ is disabled. Allocates in CurrentMemoryContext,
 * and the result is freed when that context is reset.
 */
HnswRabitq *
HnswRabitqBegin(HnswSupport * support, Datum value)
{
	HnswRabitq *rabitq;
	MemoryContextCallback *callback;
	bool		isHalf;
	RabitqMetric metric;
	Pointer		ptr;
	int			dim;
	char		errbuf[256];

	if (hnsw_rabitq_bits == 0 || DatumGetPointer(value) == NULL)
		return NULL;

	if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hnsw.rabitq_bits does not support iterative scans"),
				 errhint("Set hnsw.iterative_scan = off.")));

	if (!GetTypeAndMetric(support, &isHalf, &metric))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hnsw.rabitq_bits only supports vector and halfvec with L2, inner product, or cosine distance")));

	ptr = (Pointer) PG_DETOAST_DATUM(value);
	dim = isHalf ? ((HalfVector *) ptr)->dim : ((Vector *) ptr)->dim;

	if (dim < HNSW_RABITQ_MIN_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hnsw.rabitq_bits requires at least %d dimensions", HNSW_RABITQ_MIN_DIM)));

	rabitq = palloc0_object(HnswRabitq);
	rabitq->isHalf = isHalf;
	rabitq->dim = dim;
	if (isHalf)
		rabitq->buf = palloc(mul_size(sizeof(float), dim));

	/* Register cleanup before creating so the context can never leak */
	callback = palloc_object(MemoryContextCallback);
	callback->func = HnswRabitqFree;
	callback->arg = rabitq;
	MemoryContextRegisterResetCallback(CurrentMemoryContext, callback);

	if (rabitq_create(dim, hnsw_rabitq_bits, metric, HNSW_RABITQ_SEED, &rabitq->ctx, errbuf, sizeof(errbuf)) != 0)
		ReportError("setup", errbuf);

	if (rabitq_set_query(rabitq->ctx, GetFloats(rabitq, ptr), errbuf, sizeof(errbuf)) != 0)
		ReportError("query preparation", errbuf);

	return rabitq;
}

/*
 * Estimate the distance between the scan query and an element value
 *
 * The estimate has the same ordering as the support function (squared L2,
 * or 1 - inner product instead of negative inner product), but not the same
 * values, so it must only be compared with other estimates.
 */
double
HnswRabitqDistance(HnswRabitq * rabitq, Pointer value)
{
	float		distance;
	char		errbuf[256];

	if (rabitq_estimate(rabitq->ctx, GetFloats(rabitq, value), &distance, errbuf, sizeof(errbuf)) != 0)
		ReportError("estimation", errbuf);

	return distance;
}
