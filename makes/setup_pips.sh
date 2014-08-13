#!/bin/bash
#
# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

# $URL$
# Setup a basic pips installation from scratch

[ "$BASH_VERSION" ] || {
    echo "ERROR: script $0 to be interpreted with bash" >&2
    exit 1
}

set -eu
set -o pipefail

# where to get pips
SVN_CRI='https://scm.cri.ensmp.fr/svn'
NEWGEN_SVN="$SVN_CRI"/newgen
LINEAR_SVN="$SVN_CRI"/linear
PIPS_SVN="$SVN_CRI"/pips

# where to get polylib
POLYLIB_SITE='http://icps.u-strasbg.fr/polylib/polylib_src'
POLYLIB='polylib-5.22.5'

# minimal help
command="${0/*\//}"
usage="$command [options] [directory [developer [{checkout,export}]]]"
help="usage: $usage

positional arguments:
  directory             installation directory (default is ./MYPIPS)
  developer             login to checkout development branches (default is
                        current login)
  {checkout,export}     SVN command to run (default is checkout)

optional arguments:
  --gpips               enable gpips compilation
  -j JOBS, --jobs JOBS  number of compilation jobs to run simultaneously"

function info {
    echo
    if [ -t 1 ]; then
        echo -e "\033[1;32m### $1\033[0m"
    else
        echo "$1"
    fi
}

function error {
    if [ -t 2 ]; then
        echo -e "\033[1;31m$1\033[0m" >&2
    else
        echo "$1"
    fi
    echo "usage: $usage" >&2
    exit 1
}

function warn {
    {
        echo
        if [ -t 2 ]; then
            echo -e "\033[1;33mWARNING\033[0m"
            echo -e "\033[1;33m$1\033[0m"
            for msg in "${@:2}"; do
                echo "$msg"
            done
        else
            echo "WARNING"
            for msg in "$@"; do
                echo "$msg"
            done
        fi
    } >&2
    if [ -t 0 ]; then
        echo "Type return to continue"
        read
    fi
}

# compilation option
gpips=
full=1
numjobs=1

while [[ ${1:-} == -* ]]; do
    opt=$1
    shift
    case $opt in
    --gpips)
        gpips=1
        ;;
    --full)
        full=1
        ;;
    --light)
        full=
        ;;
    -h|--help)
        echo "$help"
        exit 0
        ;;
    -v|--version)
        echo 'version is $Id$'
        exit 0
        ;;
    -j[0-9]*)
        numjobs=$(echo $opt | cut -c 3-)
        ;;
    -j|--jobs)
        numjobs="$1"
        shift
        ;;
    -*)
        error "unexpected option $1"
        ;;
    esac
done

# arguments
destination="$(readlink -f "${1:-MYPIPS}")"
developer="${2:-${USER:-${LOGNAME:-$USERNAME}}}"
subcmd="${3:-checkout}"

# allow to substitude another make command from the environment.
make="${MAKE:-make}"
makeflags="${MAKEFLAGS:-}"

test -d "$destination"  && \
    warn "Directory $destination already exists!" \
        " If you are not trying to finish a previous installation of PIPS" \
        " in $destination you should stop and choose another directory name."
mkdir -p "$destination" || error "cannot mkdir $destination"

[ "$subcmd" = 'export' -o "$subcmd" = 'checkout' ] || \
    error "Third argument must be 'checkout' or 'export', got '$subcmd'"

prod="$destination"/prod

info "checking needed softwares"
for exe in svn wget tar gunzip "$make" cproto flex bison gcc perl sed tr ctags; do
    type "$exe" || error "no such executable, please install: $exe"
done

# check for readline
[ -d /usr/include/readline -o -d /usr/local/include/readline ] ||
    error "readline headers seem to be missing: /usr/include/readline"

# check for ncurses
[ -e /usr/include/ncurses.h -o -e /usr/local/include/ncurses.h ] ||
    error "ncurses header seems to be missing: /usr/include/ncurses.h"

# check cproto version... 4.6 is still available on many distributions
[[ $(cproto -V 2>&1) = 4.7* ]] || \
    error "Pips compilation requires at least cproto 4.7c"

# reject old svn versions because of relative externals used
[[ $(svn --version | head -1) == *' '1.[01234].* ]] &&
    error "Checking out pips requires svn 1.5 or better"

info "downloading pips"
svn "$subcmd" "$PIPS_SVN"/bundles/trunks "$prod" || error "cannot checkout pips"

if [ "$full" ]; then
    valid="$destination"/validation
    info "downloading validation"
    if ! svn "$subcmd" "$SVN_CRI"/validation/trunk "$valid"; then
        # just a warning...
        warn "cannot checkout validation"
    fi
fi

# clean environment so as not to interfere with another installation
PIPS_ARCH="$("$prod"/pips/makes/arch.sh)"
export PIPS_ARCH

# just in case
unset NEWGEN_ROOT LINEAR_ROOT PIPS_ROOT

[ "$developer" -a "$full" ] && {
    # this fails if no such developer...
    info "getting user development branches"
    svn "$subcmd" "$PIPS_SVN"/branches/"$developer" "$destination"/pips_dev || \
        info "cannot checkout PIPS development branches"
    #svn "$subcmd" "$LINEAR_SVN"/branches/"$developer" "$destination"/linear_dev
    #svn "$subcmd" "$NEWGEN_SVN"/branches/"$developer" "$destination"/newgen_dev
}

info "testing special commands for config.mk"
config="$prod"/pips/makes/config.mk
# Save an old config file if we run again this script:
[ -f "$config" ] && mv "$config" "$config".old

# falls back to "ctags -e" otherwise
type etags && echo 'ETAGS = etags' >> "$config"

type javac && echo '_HAS_JDK_ = 1' >> "$config"
type latex && echo '_HAS_LATEX_ = 1' >> "$config"
type htlatex && echo '_HAS_HTLATEX_ = 1' >> "$config"
type emacs && echo '_HAS_EMACS_ = 1' >> "$config"

# gpips is not really supported, should not be compiled?
type pkg-config && has_pkgconfig=1

if [ "$has_pkgconfig" -a "$gpips" ]; then
    echo '_HAS_PKGCONFIG_ = 1' >> "$config"
    if pkg-config --exists gtk+-2.0; then
        echo '_HAS_GTK2_ = 1' >> "$config"
    else
        echo 'PIPS_NO_GPIPS = 1' >> "$config"
    fi
else
    echo 'PIPS_NO_GPIPS = 1' >> "$config"
fi

# others? copy config to newgen and linear?
ln -s "$config" "$prod/newgen/makes/config.mk"
ln -s "$config" "$prod/linear/makes/config.mk"

# whether to build the documentation depends on latex and htlatex
target=compile
if [ "$full" ]; then
    type latex && target=build
    type htlatex && target=full-build
fi

# Find the Fortran compiler:
export PIPS_F77=
type gfortran && export PIPS_F77=gfortran
type g77 && export PIPS_F77=g77

info "creating pipsrc.sh"
cat > "$destination"/pipsrc.sh << EOF
# minimum rc file for sh-compatible shells

# default architecture is not necessary
#export PIPS_ARCH="$PIPS_ARCH"

# subversion repositories
export NEWGEN_SVN="$NEWGEN_SVN"
export LINEAR_SVN="$LINEAR_SVN"
export PIPS_SVN="$PIPS_SVN"

# production directory
prod="$prod"

# software roots are not needed
#export EXTERN_ROOT="\$prod"/extern
#export NEWGEN_ROOT="\$prod"/newgen
#export LINEAR_ROOT="\$prod"/linear
#export PIPS_ROOT="\$prod"/pips

# fix path
PATH="\$prod"/pips/bin:"\$prod"/pips/utils:"\$prod"/newgen/bin:"\$PATH"
EOF

if [ -n "$PIPS_F77" ]; then
    echo >> "$destination"/pipsrc.sh
    echo "# The Fortran compiler to use:" >> "$destination"/pipsrc.sh
    echo "export PIPS_F77=$PIPS_F77" >> "$destination"/pipsrc.sh
fi

info "generating csh environment"
"$prod"/pips/src/Scripts/env/sh2csh.pl \
    < "$destination"/pipsrc.sh \
    > "$destination"/pipsrc.csh

info "downloading $POLYLIB"
POLYLIB_TMPDIR="$(mktemp -d /tmp/polylib.XXXX)"
cd "$POLYLIB_TMPDIR" || error "cannot cd $POLYLIB_TMPDIR"
wget -nd "$POLYLIB_SITE"/"$POLYLIB".tar.gz || error "cannot wget polylib"

info "building $POLYLIB"
mkdir -p "$prod"/extern || error "cannot mkdir $prod/extern"
gunzip "$POLYLIB".tar.gz || error "cannot decompress polylib"
tar xf "$POLYLIB".tar || error "cannot untar polylib"
cd "$POLYLIB" || error "cannot cd into polylib"
./configure --prefix="$prod"/extern || error "cannot configure polylib"
# I'm not the only one to cheat with dependencies:-)
"$make" -j1 || error "cannot make polylib"

"$make" install || error "cannot install polylib"
cd /tmp || error "cannot cd /tmp"
rm -rf "$POLYLIB_TMPDIR" || error "cannot remove $POLYLIB_TMPDIR"

info "fixing $POLYLIB"
mkdir -p "$prod"/extern/lib/"$PIPS_ARCH" || error "cannot mkdir"
cd "$prod"/extern/lib/"$PIPS_ARCH" || error "cannot cd"
# Just in case a previous version was here:
rm -f libpolylib.a
ln -s ../libpolylib*.a libpolylib.a || error "cannot create links"

info "cproto header generation results in many cpp warnings..."

info "building newgen"
"$make" $makeflags -j $numjobs -C "$prod"/newgen clean "$target"

info "building linear"
"$make" $makeflags -j $numjobs -C "$prod"/linear clean "$target"

info "building pips"
# must find newgen and newC executable...
PATH="$prod"/newgen/bin:"$prod"/newgen/bin/"$PIPS_ARCH":"$PATH" \
    "$make" $makeflags -j $numjobs -C "$prod"/pips clean "$target"

info "checking for useful softwares"
# not really: wish htlatex
for exe in bash m4 latex javac emacs indent; do
    type "$exe" || echo "no such executable, consider installing: $exe"
done
