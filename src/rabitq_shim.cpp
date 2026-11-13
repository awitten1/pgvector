/*
 * Thin C wrapper around the RaBitQ reference library's stable core API
 * (rabitqlib::quant::v1). See rabitq.h for the contract.
 *
 * This file must not include any Postgres headers or call any Postgres
 * functions: Postgres reports errors with longjmp, which would skip C++
 * destructors. All errors are reported through return codes instead.
 */

#include "rabitq.h"

#include <cstdio>
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
	std::vector<float> rotatedQuery;
	std::vector<float> rotatedInput;

	/* Scratch code for rabitq_estimate */
	std::vector<uint8_t> code;

	/* Encoder workspace, aligned within workspaceStorage */
	std::vector<unsigned char> workspaceStorage;
	void	   *workspace = nullptr;

	/* Borrows rotatedQuery */
	api::QueryState queryState;
	bool		hasQuery = false;
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

static void
Encode(RabitqContext * ctx, const float *input, uint8_t *code, float *factors)
{
	ctx->rotator->rotate(input, ctx->rotatedInput.data());

	api::Factors f = api::encode(ctx->params, ctx->rotatedInput.data(), nullptr,
								 MakeCodeOutput(ctx, code),
								 api::Buffer{ctx->workspace, ctx->req.workspace_bytes});

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

		ctx->rotatedQuery.resize(padded);
		ctx->rotatedInput.resize(padded);
		ctx->code.resize(ctx->req.code_bytes + ctx->req.extra_bytes);

		size_t		alignment = ctx->req.workspace_alignment > 0 ? ctx->req.workspace_alignment : 1;
		size_t		space = ctx->req.workspace_bytes + alignment;

		ctx->workspaceStorage.resize(space);
		void	   *ptr = ctx->workspaceStorage.data();

		if (std::align(alignment, ctx->req.workspace_bytes, ptr, space) == nullptr)
			throw std::runtime_error("could not align RaBitQ workspace");
		ctx->workspace = ptr;

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
rabitq_set_query(RabitqContext * ctx, const float *query, char *errbuf, size_t errlen)
{
	return Guarded(errbuf, errlen, [&] {
		ctx->hasQuery = false;
		ctx->rotator->rotate(query, ctx->rotatedQuery.data());
		ctx->queryState = api::prepare_query(ctx->params, ctx->rotatedQuery.data(), nullptr);
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
rabitq_estimate(RabitqContext * ctx, const float *input, float *distance,
				char *errbuf, size_t errlen)
{
	return Guarded(errbuf, errlen, [&] {
		float		factors[RABITQ_NUM_FACTORS];

		Encode(ctx, input, ctx->code.data(), factors);
		*distance = Compute(ctx, ctx->code.data(), factors);
	});
}
