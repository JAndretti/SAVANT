CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -fno-math-errno -std=gnu11 -Wall -Wextra
LDFLAGS ?= -lm
OMP     ?= -fopenmp

all: cw

cw: src/cw.c
	$(CC) $(CFLAGS) $(OMP) -o $@ $< $(LDFLAGS)

# bit-reproducible build: disables FMA contraction, whose application depends
# on the compiler and on the shape of the code. Costs ~20 % at small n.
repro: src/cw.c
	$(CC) $(CFLAGS) -ffp-contract=off $(OMP) -o cw $< $(LDFLAGS)

# macOS + Apple clang: OpenMP via Homebrew's libomp (brew install libomp).
# Apple clang rejects a bare -fopenmp, and libomp is keg-only: hence
# -Xpreprocessor and the explicit paths. The prefix can be forced by hand:
# make macos OMPROOT=/opt/homebrew/opt/libomp
macos: CC := clang
macos: src/cw.c
	@P="$(OMPROOT)"; test -n "$$P" || P=$$(brew --prefix libomp 2>/dev/null); \
	 test -n "$$P" && test -f "$$P/include/omp.h" || { \
	     echo "libomp not found: brew install libomp" >&2; exit 1; }; \
	 set -x; \
	 $(CC) $(CFLAGS) -Xpreprocessor -fopenmp -I"$$P/include" \
	       -o cw $< $(LDFLAGS) -L"$$P/lib" -lomp

# build without OpenMP (macOS/clang lacking libomp: see the macos target)
serial: src/cw.c
	$(CC) $(CFLAGS) -o cw $< $(LDFLAGS)

# instrumented single-instance driver. src/trace.c #includes src/cw.c, so only
# trace.c is compiled; no OpenMP needed (one instance, one thread).
trace: src/trace.c src/cw.c
	$(CC) $(CFLAGS) -o cw_trace src/trace.c $(LDFLAGS)

debug: src/cw.c
	$(CC) -O0 -g -fsanitize=address,undefined -std=gnu11 -Wall -Wextra -o cw_dbg $< $(LDFLAGS)

clean:
	rm -f cw cw_dbg cw_trace

.PHONY: all repro macos serial trace debug clean
