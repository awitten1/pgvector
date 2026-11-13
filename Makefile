EXTENSION = vector
EXTVERSION = 0.8.7

MODULE_big = vector
DATA = $(wildcard sql/*--*--*.sql)
DATA_built = sql/$(EXTENSION)--$(EXTVERSION).sql
OBJS = src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o

# Experimental RaBitQ support (third_party/rabitqlib is a git submodule)
RABITQ_DIR = third_party/rabitqlib
RABITQ_BUILD_DIR = third_party/rabitqlib-build
RABITQ_LIB = $(RABITQ_BUILD_DIR)/librabitq_core.a
OBJS += src/hnswrabitq.o src/rabitq_shim.o
# Must match the definitions the library is built with (rabitq_compile_options)
# CPPFLAGS (not CXXFLAGS) so they also apply to LLVM bitcode
PG_CPPFLAGS += -I$(RABITQ_DIR)/include -DRABITQ_USE_OPENMP=0 -DEIGEN_MAX_ALIGN_BYTES=64 -DEIGEN_MAX_STATIC_ALIGN_BYTES=64
# No -march=native so inline Eigen code matches the library build
PG_CXXFLAGS += -std=c++17 -O2
SHLIB_LINK += $(RABITQ_LIB) -lstdc++
HEADERS = src/halfvec.h src/sparsevec.h src/vector.h

TESTS = $(wildcard test/sql/*.sql)
REGRESS = $(patsubst test/sql/%.sql,%,$(TESTS))
REGRESS_OPTS = --inputdir=test --load-extension=$(EXTENSION)

# To compile for portability, run: make OPTFLAGS=""
OPTFLAGS = -march=native

# Mac ARM doesn't always support -march=native
ifeq ($(shell uname -s), Darwin)
	ifeq ($(shell uname -p), arm)
		# no difference with -march=armv8.5-a
		OPTFLAGS =
	endif
endif

# PowerPC doesn't support -march=native
ifneq ($(filter ppc64%, $(shell uname -m)), )
	OPTFLAGS =
endif

# RISC-V64 doesn't support -march=native
ifeq ($(shell uname -m), riscv64)
	OPTFLAGS =
endif

# For auto-vectorization:
# - GCC (needs -ftree-vectorize OR -O3) - https://gcc.gnu.org/projects/tree-ssa/vectorization.html
# - Clang (could use pragma instead) - https://llvm.org/docs/Vectorizers.html
PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math -ffp-contract=fast

# Debug GCC auto-vectorization
# PG_CFLAGS += -fopt-info-vec

# Debug Clang auto-vectorization
# PG_CFLAGS += -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize

all: sql/$(EXTENSION)--$(EXTVERSION).sql

sql/$(EXTENSION)--$(EXTVERSION).sql: sql/$(EXTENSION).sql
	cp $< $@

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Postgres builds without LLVM (such as meson builds) may not configure a C++ compiler
ifeq ($(strip $(CXX)),)
CXX = c++
endif

# Build the RaBitQ core library (no ANN indexes, no OpenMP, runtime ISA dispatch)
$(RABITQ_LIB):
	@test -f $(RABITQ_DIR)/CMakeLists.txt || { echo "Run: git submodule update --init" >&2; exit 1; }
	cmake -S $(RABITQ_DIR) -B $(RABITQ_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
		-DRABITQ_BUILD_INDEX=OFF -DRABITQ_BUILD_FASTSCAN=OFF -DRABITQ_USE_OPENMP=OFF \
		-DRABITQ_BUILD_SAMPLES=OFF -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON
	cmake --build $(RABITQ_BUILD_DIR) --target rabitq_core --parallel

$(shlib): $(RABITQ_LIB)

.PHONY: rabitq-clean
rabitq-clean:
	rm -rf $(RABITQ_BUILD_DIR)

# for Mac
ifeq ($(PROVE),)
	PROVE = prove
endif

prove_installcheck:
	rm -rf $(CURDIR)/tmp_check
	cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='6$(DEF_PGPORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) $(if $(PROVE_TESTS),$(PROVE_TESTS),test/t/*.pl)

.PHONY: dist

dist:
	mkdir -p dist
	git archive --format zip --prefix=$(EXTENSION)-$(EXTVERSION)/ --output dist/$(EXTENSION)-$(EXTVERSION).zip master

# for Docker
PG_MAJOR ?= 17

.PHONY: docker

docker:
	docker build --pull --no-cache --build-arg PG_MAJOR=$(PG_MAJOR) -t pgvector/pgvector:pg$(PG_MAJOR) -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR) .

.PHONY: docker-release

docker-release:
	docker buildx build --push --pull --no-cache --platform linux/amd64,linux/arm64 --build-arg PG_MAJOR=$(PG_MAJOR) --build-arg DEBIAN_CODENAME=bookworm -t pgvector/pgvector:pg$(PG_MAJOR) -t pgvector/pgvector:pg$(PG_MAJOR)-bookworm -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR) -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR)-bookworm .

.PHONY: docker-release-trixie

docker-release-trixie:
	docker buildx build --push --pull --no-cache --platform linux/amd64,linux/arm64 --build-arg PG_MAJOR=$(PG_MAJOR) --build-arg DEBIAN_CODENAME=trixie -t pgvector/pgvector:pg$(PG_MAJOR)-trixie -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR)-trixie .
