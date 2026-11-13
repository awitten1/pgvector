/*
 * Thin C wrapper around the RaBitQ reference library's stable core API
 * (rabitqlib::quant::v1). See rabitq.h for the contract.
 *
 * This file must not include any Postgres headers or call any Postgres
 * functions: Postgres reports errors with longjmp, which would skip C++
 * destructors. All errors are reported through return codes instead.
 */

#include "rabitq.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#include <rabitqlib/core/api.hpp>
#include <rabitqlib/core/rotator.hpp>

namespace api = rabitqlib::quant::v1;

static constexpr api::CodeLayout kLayout = api::CodeLayout::SplitPacked;

struct RabitqContext
{
	api::Parameters params;
	api::Requirements req{};
	std::unique_ptr<rabitqlib::Rotator<float>> rotator;

	/* Rotated (and padded) vectors */
	std::vector<float> rotatedCentroid;
	bool		hasCentroid = false;
	std::vector<float> rotatedQuery;
	std::vector<float> rotatedInput;

	/* Scratch code for rabitq_estimate */
	std::vector<uint8_t> code;

	/* Encoder workspace, aligned within workspaceStorage */
	std::vector<unsigned char> workspaceStorage;
	void	   *workspace = nullptr;
	size_t		workspaceBytes = 0;

	/* Borrows rotatedQuery */
	api::QueryState queryState;
	bool		hasQuery = false;

	/*
	 * For rabitq_symmetric: the last first argument, reconstructed in rotated
	 * space. Kept so repeated calls with the same vector skip reconstruction.
	 */
	std::vector<uint8_t> symCode;
	std::vector<uint8_t> symUnpacked;
	std::vector<float> symQuery;
	api::QueryState symState;	/* borrows symQuery */
	float		symFactors[RABITQ_NUM_FACTORS] = {0, 0, 0};
	float		symNormSqr = 0;
	bool		hasSym = false;
};

static void
SetError(char *errbuf, size_t errlen, const char *msg) noexcept
{
	if (errbuf != nullptr && errlen > 0)
		snprintf(errbuf, errlen, "%s", msg);
}

/*
 * Run f, converting any exception into a -1 return value and an error message
 */
template <typename F>
static int
Guarded(char *errbuf, size_t errlen, F && f) noexcept
{
	try
	{
		f();
		return 0;
	}
	catch (const std::exception &e)
	{
		SetError(errbuf, errlen, e.what());
	}
	catch (...)
	{
		SetError(errbuf, errlen, "unknown C++ exception");
	}
	return -1;
}

static api::CodeOutput
MakeCodeOutput(const RabitqContext * ctx, uint8_t *code)
{
	api::CodeOutput output;

	output.layout = kLayout;
	output.codes = api::Buffer{code, ctx->req.code_bytes};
	if (ctx->req.extra_bytes > 0)
		output.extra = api::Buffer{code + ctx->req.code_bytes, ctx->req.extra_bytes};
	return output;
}

static api::CodeView
MakeCodeView(const RabitqContext * ctx, const uint8_t *code)
{
	api::CodeView view;

	view.layout = kLayout;
	view.codes = api::ConstBuffer{code, ctx->req.code_bytes};
	if (ctx->req.extra_bytes > 0)
		view.extra = api::ConstBuffer{code + ctx->req.code_bytes, ctx->req.extra_bytes};
	return view;
}

/*
 * Get the rotated centroid (nullptr for the origin)
 */
static const float *
Centroid(const RabitqContext * ctx)
{
	return ctx->hasCentroid ? ctx->rotatedCentroid.data() : nullptr;
}

static void
Encode(RabitqContext * ctx, const float *input, uint8_t *code, float *factors)
{
	const float *centroid = Centroid(ctx);
	float		normSqr = 0;

	ctx->rotator->rotate(input, ctx->rotatedInput.data());

	for (size_t i = 0; i < ctx->params.dim; i++)
	{
		float		diff = ctx->rotatedInput[i] - (centroid != nullptr ? centroid[i] : 0.0F);

		normSqr += diff * diff;
	}
	factors[3] = normSqr;

	api::Factors f = api::encode(ctx->params, ctx->rotatedInput.data(), centroid,
								 MakeCodeOutput(ctx, code),
								 api::Buffer{ctx->workspace, ctx->workspaceBytes});

	factors[0] = f.add;
	factors[1] = f.rescale;
	factors[2] = f.error;
}

static float
Compute(const RabitqContext * ctx, const uint8_t *code, const float *factors)
{
	api::Factors f;

	if (!ctx->hasQuery)
		throw std::logic_error("RaBitQ query not set");

	f.add = factors[0];
	f.rescale = factors[1];
	f.error = factors[2];
	return api::compute(ctx->queryState, MakeCodeView(ctx, code), f);
}

/*
 * Reconstruct a vector from its code for use as a query
 *
 * For a vector o with centroid c, the estimator approximates the residual
 * o - c by scale * (u - cb), with scale of -rescale / 2 for L2 (which
 * estimates |o - c|^2 + |q - c|^2 - 2 <o - c, q - c>) and -rescale for IP
 * (which estimates 1 - <o, q>). The rest is in the add factor and the terms
 * computed from the query. So o is approximated by c + scale * (u - cb).
 */
static void
SetSymmetricQuery(RabitqContext * ctx, const uint8_t *code, const float *factors)
{
	size_t		codeBytes = ctx->req.code_bytes + ctx->req.extra_bytes;

	if (ctx->hasSym && memcmp(ctx->symCode.data(), code, codeBytes) == 0 &&
		memcmp(ctx->symFactors, factors, sizeof(ctx->symFactors)) == 0)
		return;

	ctx->hasSym = false;

	api::CodeOutput unpacked;

	unpacked.layout = api::CodeLayout::UnpackedU16;
	unpacked.codes = api::Buffer{ctx->symUnpacked.data(), ctx->symUnpacked.size()};
	api::convert_codes(ctx->params, MakeCodeView(ctx, code), unpacked,
					   api::Buffer{ctx->workspace, ctx->workspaceBytes});

	size_t		dim = ctx->params.dim;
	float		cb = (static_cast<float>(1U << ctx->params.total_bits) - 1.0F) / 2.0F;
	float		scale = ctx->params.metric == rabitqlib::METRIC_L2 ? -factors[1] / 2.0F : -factors[1];
	float		normSqr = 0;
	const float *centroid = Centroid(ctx);

	for (size_t i = 0; i < dim; i++)
	{
		/* Little-endian, independent of alignment */
		unsigned	u = ctx->symUnpacked[2 * i] | (static_cast<unsigned>(ctx->symUnpacked[2 * i + 1]) << 8);
		float		v = scale * (static_cast<float>(u) - cb);

		ctx->symQuery[i] = (centroid != nullptr ? centroid[i] : 0.0F) + v;
		normSqr += v * v;
	}

	ctx->symState = api::prepare_query(ctx->params, ctx->symQuery.data(), centroid);
	ctx->symNormSqr = normSqr;
	memcpy(ctx->symCode.data(), code, codeBytes);
	memcpy(ctx->symFactors, factors, sizeof(ctx->symFactors));
	ctx->hasSym = true;
}

extern "C" int
rabitq_create(int dim, int total_bits, RabitqMetric metric, uint32_t seed,
			  RabitqContext * *out, char *errbuf, size_t errlen)
{
	*out = nullptr;

	return Guarded(errbuf, errlen, [&] {
		if (dim <= 0)
			throw std::invalid_argument("RaBitQ dimension must be positive");
		if (metric != RABITQ_METRIC_L2 && metric != RABITQ_METRIC_IP)
			throw std::invalid_argument("invalid RaBitQ metric");

		std::unique_ptr<RabitqContext> ctx(new RabitqContext());

		/* FHT-Kac requires at least 64 dimensions and pads to a multiple of 32 */
		ctx->rotator.reset(rabitqlib::choose_rotator<float>(
			static_cast<size_t>(dim), rabitqlib::RotatorType::FhtKacRotator, 0, seed));

		size_t		padded = ctx->rotator->size();

		ctx->params.dim = padded;
		ctx->params.total_bits = static_cast<size_t>(total_bits);
		ctx->params.metric = metric == RABITQ_METRIC_IP ? rabitqlib::METRIC_IP : rabitqlib::METRIC_L2;
		ctx->req = api::requirements(ctx->params, kLayout);

		ctx->rotatedCentroid.resize(padded);
		ctx->rotatedQuery.resize(padded);
		ctx->rotatedInput.resize(padded);
		ctx->code.resize(ctx->req.code_bytes + ctx->req.extra_bytes);

		/* The workspace is shared by encoding and code conversion */
		api::Requirements unpackedReq = api::requirements(ctx->params, api::CodeLayout::UnpackedU16);
		size_t		alignment = std::max<size_t>({ctx->req.workspace_alignment, unpackedReq.workspace_alignment, 1});

		ctx->workspaceBytes = std::max(ctx->req.workspace_bytes, unpackedReq.workspace_bytes);

		size_t		space = ctx->workspaceBytes + alignment;

		ctx->workspaceStorage.resize(space);
		void	   *ptr = ctx->workspaceStorage.data();

		if (std::align(alignment, ctx->workspaceBytes, ptr, space) == nullptr)
			throw std::runtime_error("could not align RaBitQ workspace");
		ctx->workspace = ptr;

		ctx->symCode.resize(ctx->req.code_bytes + ctx->req.extra_bytes);
		ctx->symUnpacked.resize(unpackedReq.code_bytes);
		ctx->symQuery.resize(padded);

		*out = ctx.release();
	});
}

extern "C" void
rabitq_free(RabitqContext * ctx)
{
	delete ctx;
}

extern "C" size_t
rabitq_code_bytes(const RabitqContext * ctx)
{
	return ctx->req.code_bytes + ctx->req.extra_bytes;
}

extern "C" int
rabitq_set_centroid(RabitqContext * ctx, const float *centroid, char *errbuf, size_t errlen)
{
	return Guarded(errbuf, errlen, [&] {
		ctx->hasCentroid = false;
		ctx->hasQuery = false;
		ctx->hasSym = false;
		ctx->rotator->rotate(centroid, ctx->rotatedCentroid.data());
		ctx->hasCentroid = true;
	});
}

extern "C" int
rabitq_set_query(RabitqContext * ctx, const float *query, char *errbuf, size_t errlen)
{
	return Guarded(errbuf, errlen, [&] {
		ctx->hasQuery = false;
		ctx->rotator->rotate(query, ctx->rotatedQuery.data());
		ctx->queryState = api::prepare_query(ctx->params, ctx->rotatedQuery.data(), Centroid(ctx));
		ctx->hasQuery = true;
	});
}

extern "C" int
rabitq_encode(RabitqContext * ctx, const float *input, uint8_t *code,
			  float factors[RABITQ_NUM_FACTORS], char *errbuf, size_t errlen)
{
	return Guarded(errbuf, errlen, [&] {
		Encode(ctx, input, code, factors);
	});
}

extern "C" int
rabitq_compute(const RabitqContext * ctx, const uint8_t *code,
			   const float factors[RABITQ_NUM_FACTORS], float *distance,
			   char *errbuf, size_t errlen)
{
	return Guarded(errbuf, errlen, [&] {
		*distance = Compute(ctx, code, factors);
	});
}

extern "C" int
rabitq_symmetric(RabitqContext * ctx, const uint8_t *code_a,
				 const float factors_a[RABITQ_NUM_FACTORS], const uint8_t *code_b,
				 const float factors_b[RABITQ_NUM_FACTORS], float *distance,
				 char *errbuf, size_t errlen)
{
	return Guarded(errbuf, errlen, [&] {
		api::Factors f;

		SetSymmetricQuery(ctx, code_a, factors_a);

		f.add = factors_b[0];
		f.rescale = factors_b[1];
		f.error = factors_b[2];

		float		d = api::compute(ctx->symState, MakeCodeView(ctx, code_b), f);

		if (ctx->params.metric == rabitqlib::METRIC_L2)
		{
			/*
			 * The reconstructed residual is longer than the original, so use
			 * the original squared distance to the centroid for the query
			 * term. Errors of similar vectors are correlated, which
			 * underestimates their distance, so apply the triangle
			 * inequality (through the centroid) as a lower bound.
			 */
			float		normA = std::sqrt(std::max(factors_a[3], 0.0F));
			float		normB = std::sqrt(std::max(factors_b[3], 0.0F));

			d = d - ctx->symNormSqr + factors_a[3];
			d = std::max(d, (normA - normB) * (normA - normB));
		}
		else
		{
			/*
			 * The estimate includes <q, c>, which used the reconstructed
			 * residual r' in place of the original residual r. The add
			 * factor is 1 - <r, c> + <r', c>, so this replaces <r', c> with
			 * the exact <r, c>. No effect when the centroid is the origin
			 * (add is 1).
			 */
			d += factors_a[0] - 1.0F;
		}

		*distance = d;
	});
}

extern "C" int
rabitq_estimate(RabitqContext * ctx, const float *input, float *distance,
				char *errbuf, size_t errlen)
{
	return Guarded(errbuf, errlen, [&] {
		float		factors[RABITQ_NUM_FACTORS];

		Encode(ctx, input, ctx->code.data(), factors);
		*distance = Compute(ctx, ctx->code.data(), factors);
	});
}
