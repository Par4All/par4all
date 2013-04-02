#! /bin/bash
#
# $Id$
#
# Copyright 1989-2012 MINES ParisTech
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

# where to get pips
SVN_CRI='https://svn.cri.ensmp.fr/svn'
PIPS_SVN=$SVN_CRI/pips

#POLYLIB_SITE='http://www.cri.ensmp.fr/pips'
POLYLIB_SITE='http://icps.u-strasbg.fr/polylib/polylib_src'
POLYLIB='polylib-5.22.5'

# help
command=${0/*\//}
usage="usage: $command [destination-directory [developer [checkout|export]]]"

# arguments
destination=${1:-`pwd`/MYPIPS}
developer=${2:-${USER:-${LOGNAME:-$USERNAME}}}
subcmd=${3:-checkout}

# allow to substitude another make command from the environment.
make=${MAKE:-make}

# If the destination directory is relative, transform it in an absolute
# path name:
[[ $destination != /* ]] && destination=`pwd`/$destination

error()
{
  echo "$@" >&2
  exit 1
}

warn()
{
  {
    echo
    echo "WARNING"
    for msg in "$@" ; do
      echo $msg ;
    done
    echo "Type return to continue"
  } >&2
  read
}

test -d $destination  && \
    warn "Directory $destination already exists!" \
      " If you are not trying to finish a previous installation of PIPS" \
      " in $destination you should stop and choose another directory name."
mkdir -p $destination || error "cannot mkdir $destination"

[ $subcmd = 'export' -o $subcmd = 'checkout' ] || \
    error "Third argument must be 'checkout' or 'export', got '$subcmd'"

prod=$destination/prod

echo
echo "### checking needed softwares"
# ed is used within the "newgen" script
for exe in svn wget tar gunzip $make cproto flex bison gcc perl sed tr ed
do
  type $exe || error "no such executable, please install: $exe"
done

# check for readline
[ -d /usr/include/readline -o -d /usr/local/include/readline ] ||
  error "readline headers seem to be missing: /usr/include/readline"

# check cproto version... 4.6 is still available on many distributions
[[ $(cproto -V 2>&1) = 4.7* ]] || \
  error "Pips compilation requires at least cproto 4.7c"

# reject old svn versions because of relative externals used
[[ $(svn --version | head -1) == *' '1.[01234].* ]] &&
  error "Checking out pips requires svn 1.5 or better"

echo
echo "### downloading pips"
svn $subcmd $PIPS_SVN/bundles/trunks $prod || error "cannot checkout pips"

valid=$destination/validation
echo "### downloading validation"
if ! svn $subcmd $SVN_CRI/validation/trunk $valid
then
  # just a warning...
  warn "cannot checkout validation"
fi

# clean environment so as not to interfere with another installation
PIPS_ARCH=`$prod/pips/makes/arch.sh`
export PIPS_ARCH
unset NEWGEN_ROOT LINEAR_ROOT PIPS_ROOT

[ "$developer" ] &&
{
  # this fails if no such developer...
  echo "### getting user development branches"
  svn $subcmd $PIPS_SVN/branches/$developer $destination/pips_dev
  #svn $subcmd $LINEAR_SVN/branches/$developer $destination/linear_dev
  #svn $subcmd $NEWGEN_SVN/branches/$developer $destination/newgen_dev
}

echo
echo "### testing special commands for config.mk"
config=$prod/pips/makes/config.mk
# Save an old config file if we run again this script:
[ -f $config ] && mv $config $config.old

type javac && echo '_HAS_JDK_ = 1' >> $config
type latex && echo '_HAS_LATEX_ = 1' >> $config
type htlatex && echo '_HAS_HTLATEX_ = 1' >> $config
type emacs && echo '_HAS_EMACS_ = 1' >> $config
type pkg-config && has_pkgconfig=1

if [ "$has_pkgconfig" ]
then
  echo '_HAS_PKGCONFIG_ = 1' >> $config
  if pkg-config --exists gtk+-2.0
  then
    echo '_HAS_GTK2_ = 1' >> $config
 else
    echo 'PIPS_NO_GPIPS = 1' >> $config
  fi
else
  echo 'PIPS_NO_GPIPS = 1' >> $config
fi

# others? copy config to newgen and linear?
ln -s $config $prod/newgen/makes/config.mk
ln -s $config $prod/linear/makes/config.mk

# whether to build the documentation depends on latex and htlatex
target=compile
type latex && target=build
type htlatex && target=full-build

# Find the Fortran compiler:
type gfortran && export PIPS_F77=gfortran
type g77 && export PIPS_F77=g77

echo
echo "### creating pipsrc.sh"
cat <<EOF > $destination/pipsrc.sh
# minimum rc file for sh-compatible shells

# default architecture is not necessary
#export PIPS_ARCH=$PIPS_ARCH

# subversion repositories
export NEWGEN_SVN=$SVN_CRI/newgen
export LINEAR_SVN=$SVN_CRI/linear
export PIPS_SVN=$SVN_CRI/pips

# production directory
prod=$prod

# software roots are not needed
#export EXTERN_ROOT=\$prod/extern
#export NEWGEN_ROOT=\$prod/newgen
#export LINEAR_ROOT=\$prod/linear
#export PIPS_ROOT=\$prod/pips

# fix path
PATH=\$prod/pips/bin:\$prod/pips/utils:\$prod/newgen/bin:\$PATH
EOF

if [ -n "$PIPS_F77" ]; then
    echo >> $destination/pipsrc.sh
    echo "# The Fortran compiler to use:" >> $destination/pipsrc.sh
    echo "export PIPS_F77=$PIPS_F77" >> $destination/pipsrc.sh
fi

echo
echo "### generating csh environment"
$prod/pips/src/Scripts/env/sh2csh.pl \
    < $destination/pipsrc.sh \
    > $destination/pipsrc.csh

echo
echo "### downloading $POLYLIB"
cd /tmp
if test -f $POLYLIB.tar.gz
then
  warn "some /tmp/$POLYLIB.tar.gz file already there. Continue using it?"
else
  wget -nd $POLYLIB_SITE/$POLYLIB.tar.gz || error "cannot wget polylib"
fi

echo
echo "### building $POLYLIB"
mkdir -p $prod/extern || error "cannot mkdir $prod/extern"
gunzip $POLYLIB.tar.gz || error "cannot decompress polylib"
tar xf $POLYLIB.tar || error "cannot untar polylib"
cd $POLYLIB || error "cannot cd into polylib"
./configure --prefix=$prod/extern || error "cannot configure polylib"
# I'm not the only one to cheat with dependencies:-)
$make -j1 || error "cannot make polylib"

$make install || error "cannot install polylib"
cd .. || error "cannot cd .."
rm -rf $POLYLIB || error "cannot remove polylib"
rm -f $POLYLIB.tar || error "cannot remove polylib tar"

echo
echo "### fixing $POLYLIB"
mkdir -p $prod/extern/lib/$PIPS_ARCH || error "cannot mkdir"
cd $prod/extern/lib/$PIPS_ARCH || error "cannot cd"
# Just in case a previous version was here:
rm -f libpolylib.a
ln -s ../libpolylib*.a libpolylib.a || error "cannot create links"

warn "cproto header generation results in many cpp warnings..."

echo
echo "### building newgen"
cd $prod/newgen
$make clean
$make $target

echo
echo "### building linear"
cd $prod/linear
$make clean
$make $target

echo
echo "### building pips"
cd $prod/pips
$make clean
# must find newgen and newC executable...
PATH=$prod/newgen/bin:$prod/newgen/bin/$PIPS_ARCH:$PATH \
    $make $target

echo
echo "### checking for useful softwares"
# not really: wish htlatex
for exe in bash m4 latex javac emacs indent
do
  type $exe || echo "no such executable, consider installing: $exe"
done
