#!/bin/bash
set -eu
set -o pipefail

progname="$(basename "$0")"
args=$(getopt --name "$progname" --long "help,pipsdir:,distdir:" --options "hp:d:" -- "$@")
usage="Usage: $progname [-h] [-p PIPSDIR] [-a PIPSARCH]

Options:
  -h, --help            show this help message and exist
  -p PIPSDIR, --pipsdir PIPSDIR
                        PIPS directory
  -a PIPSARCH, --pipsarch PIPSARCH
                        PIPS architecture"

eval set -- "$args"
while [ $# -gt 0 ]; do
    case "$1" in
        "-h" | "--help")     echo "$usage"; exit;;
        "-p" | "--pipsdir")  pipsdir="$2";;
        "-a" | "--pipsarch") pipsarch="$2";;
        "--")                shift; break;;
    esac
    shift
done

basedir="$(readlink -f "$(dirname "$0")")"

function die {
    echo "$progname: ${1:-"Unknown error"}" 1>&2
    exit 1
}

pipsdir="$(readlink -f "${pipsdir:-"$basedir/../../.."}")"
pipsarch="${pipsarch:-LINUX_x86_64_LL}"
pipsrev=$(svnversion "$pipsdir")
pipsbin="$pipsdir/bin/$pipsarch/fpips"
pipsdoc="$pipsdir/src/Documentation"

function check_pipsbin {
    [ -x "$pipsbin" ] || \
        die "Cannot find fpips executable in '$(dirname "$pipsbin")'"
    local libs="$(objdump -p "$pipsbin" | \
        sed -n "s/ *NEEDED \+\([^\.]\+\).*/\1/p" | sort | paste -sd ",")"
    [ "$libs" = "libc,libm,libncursesw,libreadline" ] || \
        die "Something is wrong in fpips library dependencies"
}
check_pipsbin

function check_pipsdoc {
    if [ ! -e "$pipsdoc/dev_guide/developer_guide.pdf" ] || \
        [ ! -e "$pipsdoc/tpips-user-manual/tpips-user-manual.pdf" ]; then
        die "Documentation is missing"
    fi
}
check_pipsdoc

tmpdir="$(mktemp -p /tmp -d pips_tar_XXXX)"
trap "rm -rf $tmpdir" EXIT

distname="pips_r$pipsrev"
distdir="$tmpdir/$distname"
mkdir -p "$distdir"

function copy_license {
    cp "$pipsdir/COPYING" "$distdir/LICENSE"
}
copy_license

function copy_readme {
    cp "$basedir/pips_tar_readme" "$distdir/README"
}
copy_readme

function copy_pipsbin {
    mkdir -p "$distdir/bin"
    cp "$pipsbin" "$distdir/bin/"
    ln -s fpips "$distdir/bin/pips"
    ln -s fpips "$distdir/bin/tpips"
}
copy_pipsbin

function copy_pipsdoc {
    mkdir -p "$distdir/doc"
    cp "$pipsdoc/dev_guide/developer_guide.pdf" "$distdir/doc/"
    cp "$pipsdoc/tpips-user-manual/tpips-user-manual.pdf" "$distdir/doc/"
}
copy_pipsdoc

function download_tests {
    mkdir -p "$distdir/tests"
    svn checkout -q "https://svn.cri.ensmp.fr/svn/validation/trunk/Demo/TutorialPPoPP2010.sub/" "$distdir/tests"
    rm -rf "$distdir/tests/.svn" "$distdir/tests/Makefile"
}
download_tests

testdir="$tmpdir/test"
mkdir -p "$testdir"

function run_test {
    cat << EOF > "$testdir/foo.c"
int main(void) {
    int i = 3;
    i -= 3;
    return 0;
}
EOF
    cat << EOF > "$testdir/foo.tpips"
create foo foo.c
activate PRINT_CODE_PRECONDITIONS
display PRINTED_FILE
close
delete foo
EOF
    (cd "$testdir" && "$distdir/bin/tpips" "foo.tpips" > "out" 2> /dev/null) || \
        die "Cannot run tpips"
    grep -q "{i==0}" "$testdir/out" || die "Precondition not found"
}
run_test

function create_tarball {
    tar -C "$tmpdir" -zcf "$basedir/$distname.tar.gz" "$distname/"
}
create_tarball

echo "OK"
