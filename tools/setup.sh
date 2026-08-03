#!/bin/sh
# setup.sh — check the toolchain, then fetch the datasets and build every
# reference solver. Idempotent: anything already present is left alone, so it
# is safe to re-run.
#
#   sh tools/setup.sh              # everything
#   sh tools/setup.sh check        # only report what is missing
#   sh tools/setup.sh data         # only the instance sets
#   sh tools/setup.sh solvers      # only SAVANT + HGS + LKH-3 + AILS-II
#
# AILS-II is the only part needing a JDK. If you do not want one, everything
# else still works; the script says so and carries on.
set -e
cd "$(dirname "$0")/.."
STEP=${1:-all}

say()  { printf '\n=== %s\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- requirements
missing=""
need() {
    if have "$1"; then
        printf '  %-10s %s\n' "$1" "ok"
    else
        printf '  %-10s MISSING  (%s)\n' "$1" "$2"
        missing="$missing $1"
    fi
}

say "requirements"
need cc      "any C compiler; on macOS install the Xcode command line tools"
need make    "build SAVANT and LKH-3"
need cmake   "build HGS-CVRP"
need curl    "download the instance sets and LKH-3"
need bsdtar  "reads .7z natively; Debian/Ubuntu: apt install libarchive-tools"
need git     "clone HGS-CVRP and AILS-II"
need python3 "the tools/ scripts"
need uv      "how the tools/ scripts are invoked here; plain python3 also works"

# OpenMP: SAVANT is parallel. Apple clang needs Homebrew's libomp and a
# different flag, which is what `make macos` is for.
OMP_TARGET=all
if [ "$(uname -s)" = "Darwin" ]; then
    if brew --prefix libomp >/dev/null 2>&1; then
        printf '  %-10s %s\n' "libomp" "ok (make macos)"
        OMP_TARGET=macos
    else
        printf '  %-10s %s\n' "libomp" "MISSING -> building SAVANT single-threaded"
        printf '             install with: brew install libomp\n'
        OMP_TARGET=serial
    fi
fi

# Java is optional: only AILS-II needs it.
JAVA=""
for c in /opt/homebrew/opt/openjdk/bin/java /usr/local/opt/openjdk/bin/java java; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -version >/dev/null 2>&1; then
        JAVA=$c; break
    fi
done
if [ -n "$JAVA" ]; then
    printf '  %-10s %s\n' "java" "ok ($JAVA)"
else
    printf '  %-10s %s\n' "java" "MISSING -> AILS-II will be skipped"
    printf '             install with: brew install openjdk   (or apt install default-jdk)\n'
fi

if [ -n "$missing" ]; then
    echo
    echo "missing required tools:$missing" >&2
    [ "$STEP" = check ] || exit 1
fi
[ "$STEP" = check ] && exit 0

# ------------------------------------------------------------------- datasets
if [ "$STEP" = all ] || [ "$STEP" = data ]; then
    say "instance sets"
    # NeuOpt: 10000 instances each at n = 20/50/100 (+200). CONTINUOUS
    # coordinates in [0,1]: these must NOT be rounded.
    if [ -f data/cvrp_100.cvrpb ]; then
        echo "  NeuOpt sets already in data/"
    else
        uv run --no-project tools/fetch_neuopt.py
    fi
    # CVRPLib X / XL / XML100. ROUNDED (TSPLIB EUC_2D), and they ship the
    # optima or BKS that everything is scored against.
    uv run --no-project tools/fetch_cvrplib.py --verify
    # The XL archive holds the PRE-challenge BKS; the paper has the final ones.
    if ls paper/*.tex >/dev/null 2>&1; then
        uv run --no-project tools/parse_xl_bks.py --check
    else
        echo "  paper/*.tex not present -> skipping baseline/xl_bks.csv"
        echo "  (XL will be scored against the pre-challenge .sol files)"
    fi
fi

# -------------------------------------------------------------------- solvers
if [ "$STEP" = all ] || [ "$STEP" = solvers ]; then
    mkdir -p external

    say "SAVANT"
    make "$OMP_TARGET"
    ./cw --help >/dev/null && echo "  ./cw ok"

    say "HGS-CVRP (Vidal 2022)"
    [ -d external/HGS-CVRP ] || \
        git clone --depth 1 https://github.com/vidalt/HGS-CVRP.git external/HGS-CVRP
    cmake -S external/HGS-CVRP -B external/HGS-CVRP/build \
          -DCMAKE_BUILD_TYPE=Release >/dev/null
    make -C external/HGS-CVRP/build bin >/dev/null
    echo "  external/HGS-CVRP/build/hgs ok"

    say "LKH-3 (Helsgaun)"
    if [ ! -x external/LKH-3.0.14/LKH ]; then
        [ -f external/LKH-3.0.14.tgz ] || curl -sL --fail -o external/LKH-3.0.14.tgz \
            "http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz"
        tar xzf external/LKH-3.0.14.tgz -C external/
        make -C external/LKH-3.0.14 >/dev/null 2>&1
    fi
    echo "  external/LKH-3.0.14/LKH ok"

    say "AILS-II (Maximo, Cordeau, Nascimento)"
    if [ -z "$JAVA" ]; then
        echo "  skipped: no JDK (brew install openjdk)"
    else
        [ -d external/AILS-II ] || \
            git clone --depth 1 https://github.com/INFORMSJoC/2023.0106.git external/AILS-II
        # The upstream code prints only a cost line. The patch adds
        # -solution <file> so validate.py can check its routes; main() only.
        if ! grep -q '"-solution"' external/AILS-II/src/SearchMethod/AILSII.java; then
            patch -p0 -d external/AILS-II < tools/ails_solution_output.patch
        fi
        JAVAC=$(dirname "$JAVA")/javac
        JAR=$(dirname "$JAVA")/jar
        mkdir -p external/AILS-II/build
        find external/AILS-II/src -name '*.java' > external/AILS-II/srcs.txt
        "$JAVAC" -nowarn -d external/AILS-II/build @external/AILS-II/srcs.txt
        (cd external/AILS-II/build && "$JAR" cfe ../AILSII.jar SearchMethod.AILSII .)
        rm -f external/AILS-II/srcs.txt
        echo "  external/AILS-II/AILSII.jar ok"
    fi
fi

say "done"
cat <<'EOF'
Next:
  sh tools/compare_all.sh X 24 10      # all four solvers, 24 instances, 10 s each
  sh tools/compare_hgs.sh 1000         # SAVANT vs HGS on the NeuOpt CVRP-100 set
EOF
