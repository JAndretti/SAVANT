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
# POSIX sh, no bashisms, no `uname` branching except where a platform genuinely
# differs: everything is decided by probing for the tool, not by guessing from
# the OS name. Where a dependency is missing the script prints the install line
# for every common package manager, and — for cmake, the .7z reader and the JDK
# — offers a route that needs no root at all, since a benchmark harness should
# not require administrator rights on the machine it is measuring.
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
        printf '  %-10s MISSING\n' "$1"
        printf '             %s\n' "$2"
        missing="$missing $1"
    fi
}

say "requirements"
need cc   "a C compiler, for SAVANT and LKH-3.
             Debian/Ubuntu: apt install build-essential
             Fedora/RHEL:   dnf install gcc make
             Arch:          pacman -S base-devel
             macOS:         xcode-select --install"
need make "usually arrives with the C compiler above"
need curl "downloads the instance sets and LKH-3; ships with macOS.
             Debian/Ubuntu: apt install curl"
need git  "clones HGS-CVRP and AILS-II.
             Debian/Ubuntu: apt install git"
need python3 "runs the tools/ scripts.
             Debian/Ubuntu: apt install python3"
need uv   "how the tools/ scripts are invoked here; plain python3 also works.
             any OS, no root: curl -LsSf https://astral.sh/uv/install.sh | sh"

# cmake builds HGS-CVRP. There is a pip/uv wheel for every platform, so a
# missing cmake never needs root.
if have cmake; then
    printf '  %-10s %s\n' "cmake" "ok"
else
    printf '  %-10s MISSING\n' "cmake"
    printf '             builds HGS-CVRP.
             Debian/Ubuntu: apt install cmake
             Fedora/RHEL:   dnf install cmake
             Arch:          pacman -S cmake
             macOS:         brew install cmake
             any OS, no root: uv tool install cmake\n'
    missing="$missing cmake"
fi

# The CVRPLib archives are .7z. Any one of bsdtar, the 7-Zip CLI, or the
# pure-Python py7zr will do; fetch_cvrplib.py tries them in that order. py7zr
# is pulled in on the fly by `uv run --with`, so it needs no root and does not
# become a permanent dependency of the project.
SEVENZIP=""
for c in bsdtar 7zz 7z 7za 7zr; do
    if have "$c"; then SEVENZIP=$c; break; fi
done
UVRUN="uv run --no-project"
if [ -n "$SEVENZIP" ]; then
    printf '  %-10s %s\n' ".7z" "ok ($SEVENZIP)"
elif have uv; then
    printf '  %-10s %s\n' ".7z" "no system reader -> using py7zr via uv"
    UVRUN="uv run --no-project --with py7zr"
else
    printf '  %-10s MISSING\n' ".7z"
    printf '             reads the CVRPLib archives.
             Debian/Ubuntu: apt install libarchive-tools   (bsdtar)
             Fedora/RHEL:   dnf install bsdtar
             Arch:          pacman -S libarchive
             macOS:         already there
             any OS, no root: pip install py7zr\n'
    missing="$missing 7z-reader"
fi

# Java is optional: only AILS-II needs it, and it needs a JDK (javac + jar),
# not just the JRE that `java` alone indicates.
JAVAC=""
for c in external/jdk/bin/javac "${JAVA_HOME:-/nonexistent}/bin/javac" javac \
         /opt/homebrew/opt/openjdk/bin/javac /usr/local/opt/openjdk/bin/javac; do
    case $c in
        /*|./*|external/*) [ -x "$c" ] || continue ;;
        *) have "$c" || continue ;;
    esac
    JAVAC=$c; break
done
if [ -z "$JAVAC" ]; then
    for c in /usr/lib/jvm/*/bin/javac /usr/java/*/bin/javac; do
        [ -x "$c" ] && JAVAC=$c
    done
fi
if [ -n "$JAVAC" ]; then
    # Absolute, because the jar step runs from inside external/AILS-II/build.
    case $JAVAC in
        /*) ;;
        */*) JAVAC=$PWD/$JAVAC ;;
        *) JAVAC=$(command -v "$JAVAC") ;;
    esac
    printf '  %-10s %s\n' "javac" "ok ($JAVAC)"
else
    printf '  %-10s %s\n' "javac" "MISSING -> AILS-II will be skipped"
    printf '             a JDK, not just a JRE: `java` alone is not enough.
             Debian/Ubuntu: apt install default-jdk
             Fedora/RHEL:   dnf install java-latest-openjdk-devel
             Arch:          pacman -S jdk-openjdk
             macOS:         brew install openjdk
             any OS, no root: sh tools/setup.sh jdk   (unpacks one in external/jdk)\n'
fi

if [ -n "$missing" ]; then
    echo
    echo "missing required tools:$missing" >&2
    [ "$STEP" = check ] || exit 1
fi
[ "$STEP" = check ] && exit 0

# ------------------------------------------------------ a local JDK, no root
# Fetches an Eclipse Temurin JDK into external/jdk. Used when the machine has
# no JDK and no way to install one system-wide; run_ails.py looks there first.
if [ "$STEP" = jdk ]; then
    say "Eclipse Temurin JDK 21 -> external/jdk"
    if [ -x external/jdk/bin/javac ]; then
        echo "  already there"
        exit 0
    fi
    case $(uname -s) in
        Darwin) OS=mac ;;
        Linux)  OS=linux ;;
        *)      echo "unsupported platform $(uname -s); install a JDK by hand" >&2
                exit 1 ;;
    esac
    case $(uname -m) in
        x86_64|amd64)  ARCH=x64 ;;
        arm64|aarch64) ARCH=aarch64 ;;
        *) echo "unsupported architecture $(uname -m); install a JDK by hand" >&2
           exit 1 ;;
    esac
    mkdir -p external
    curl -L --fail -o external/jdk.tar.gz \
        "https://api.adoptium.net/v3/binary/latest/21/ga/$OS/$ARCH/jdk/hotspot/normal/eclipse"
    mkdir -p external/jdk
    tar xzf external/jdk.tar.gz -C external/jdk --strip-components=1
    rm -f external/jdk.tar.gz
    external/jdk/bin/javac -version
    echo "  external/jdk ok — now run: sh tools/setup.sh solvers"
    exit 0
fi

# ------------------------------------------------------------------- datasets
if [ "$STEP" = all ] || [ "$STEP" = data ]; then
    say "instance sets"
    # NeuOpt: 10000 instances each at n = 20/50/100 (+200). CONTINUOUS
    # coordinates in [0,1]: these must NOT be rounded.
    if [ -f data/cvrp_100.cvrpb ]; then
        echo "  NeuOpt sets already in data/"
    else
        $UVRUN tools/fetch_neuopt.py
    fi
    # tests/fuzz.py exercises cw's --dir path on data/cvrp_200/, which only the
    # --tsplib option writes; 12 MB, and without it 10 of its 200 trials fail
    # on a missing directory rather than on anything about the solver.
    if [ -d data/cvrp_200 ]; then
        echo "  data/cvrp_200/ (TSPLIB, for tests/fuzz.py) already there"
    else
        $UVRUN tools/fetch_neuopt.py --sizes 200 --tsplib
    fi
    # CVRPLib X / XL / XML100. ROUNDED (TSPLIB EUC_2D), and they ship the
    # optima or BKS that everything is scored against.
    $UVRUN tools/fetch_cvrplib.py --verify
    # The XL archive holds the PRE-challenge BKS; the paper has the final ones.
    if ls paper/*.tex >/dev/null 2>&1; then
        $UVRUN tools/parse_xl_bks.py --check
    else
        echo "  paper/*.tex not present -> skipping baseline/xl_bks.csv"
        echo "  (XL will be scored against the pre-challenge .sol files)"
    fi
fi

# -------------------------------------------------------------------- solvers
if [ "$STEP" = all ] || [ "$STEP" = solvers ]; then
    mkdir -p external

    say "SAVANT"
    # `make auto` probes for OpenMP instead of guessing from the platform, and
    # falls back to a single-threaded build rather than failing.
    make auto
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
    if [ -z "$JAVAC" ]; then
        echo "  skipped: no JDK. Install one, or: sh tools/setup.sh jdk"
    else
        [ -d external/AILS-II ] || \
            git clone --depth 1 https://github.com/INFORMSJoC/2023.0106.git external/AILS-II
        # The upstream code prints only a cost line. The patch adds
        # -solution <file> so validate.py can check its routes; main() only.
        if ! grep -q '"-solution"' external/AILS-II/src/SearchMethod/AILSII.java; then
            patch -p0 -d external/AILS-II < tools/ails_solution_output.patch
        fi
        JAR=$(dirname "$JAVAC")/jar
        mkdir -p external/AILS-II/build
        # @argfile rather than $(find ...): the command line stays short, and
        # nothing depends on the shell's word splitting.
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
