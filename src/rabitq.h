#ifndef RABITQ_H
#define RABITQ_H

/*
 * C interface to the RaBitQ reference library (third_party/rabitqlib).
 *
 * The implementation (rabitq_shim.cpp) is C++, but nothing C++ crosses this
 * boundary: every function catches all exceptions and reports failure with a
 * -1 return value and a message in errbuf. Callers are expected to turn that
 * into an ereport. The shim never calls into Postgres (no palloc, no elog),
 * so a longjmp can never unwind through C++ frames.
 *
 * Memory owned by a RabitqContext comes from the C++ heap, is allocated only
 * in rabitq_create, and must be released with rabitq_free. The per-vector
 * calls (set_query, encode, compute, estimate) do not allocate.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct RabitqContext RabitqContext;

typedef enum RabitqMetric
{
	RABITQ_METRIC_L2 = 0,		/* estimates squared L2 distance */
	RABITQ_METRIC_IP = 1		/* estimates 1 - inner product */
}			RabitqMetric;

/*
 * Number of floats in a factors array: the three factors from the library,
 * followed by the squared distance to the centroid (for rabitq_symmetric)
 */
#define RABITQ_NUM_FACTORS 4

/*
 * Create a context for vectors with dim dimensions (at least 64) quantized
 * with total_bits bits per dimension (1..9). Rotation is a seeded FHT-Kac
 * rotation, so the same seed always gives the same rotation.
 */
int			rabitq_create(int dim, int total_bits, RabitqMetric metric, uint32_t seed,
						  RabitqContext * *out, char *errbuf, size_t errlen);

/* Free a context; NULL is allowed */
void		rabitq_free(RabitqContext * ctx);

/* Size of one encoded vector in bytes, excluding factors */
size_t		rabitq_code_bytes(const RabitqContext * ctx);

/*
 * Set the centroid that vectors are encoded relative to (dim floats).
 * Must be called before any other function, and codes are only comparable
 * when they use the same centroid. The default is the origin.
 */
int			rabitq_set_centroid(RabitqContext * ctx, const float *centroid,
								char *errbuf, size_t errlen);

/* Set the query used by rabitq_compute and rabitq_estimate */
int			rabitq_set_query(RabitqContext * ctx, const float *query,
							 char *errbuf, size_t errlen);

/* Encode input into code (rabitq_code_bytes bytes) and factors */
int			rabitq_encode(RabitqContext * ctx, const float *input, uint8_t *code,
						  float factors[RABITQ_NUM_FACTORS], char *errbuf, size_t errlen);

/* Estimate the distance between the query and an encoded vector */
int			rabitq_compute(const RabitqContext * ctx, const uint8_t *code,
						   const float factors[RABITQ_NUM_FACTORS], float *distance,
						   char *errbuf, size_t errlen);

/*
 * Estimate the distance between two encoded vectors. The first is
 * reconstructed from its code and used as the query for the second, so the
 * estimate is less accurate than rabitq_compute.
 */
int			rabitq_symmetric(RabitqContext * ctx, const uint8_t *code_a,
							 const float factors_a[RABITQ_NUM_FACTORS],
							 const uint8_t *code_b,
							 const float factors_b[RABITQ_NUM_FACTORS],
							 float *distance, char *errbuf, size_t errlen);

/* Encode input and estimate its distance to the query in one step */
int			rabitq_estimate(RabitqContext * ctx, const float *input, float *distance,
							char *errbuf, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
