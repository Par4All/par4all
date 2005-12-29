#! /bin/sh
#
# $Id$
#
# Setup a pips installation from scratch
#

# where to get pips
SVN_CRI='http://svn.cri.ensmp.fr/svn'
PIPS_SVN=$SVN_CRI/pips

#POLYLIB_SITE='http://icps.u-strasbg.fr/polylib/polylib_src'
POLYLIB_SITE='http://www.cri.ensmp.fr/pips'
POLYLIB='polylib-5.22.1'

# help
command=${0/*\//}
usage="usage: $command [destination-directory [developer]]"

# arguments
destination=${1:-`pwd`/MYPIPS}
developer=${2:-$USERNAME} 

error()
{
    echo "$@" >&2
    exit 1
}

test -d $destination && error "directory $destination already exists"
mkdir $destination || error "cannot mkdir $destination"

prod=$destination/prod

echo "### checking needed softwares"
for exe in svn wget tar gunzip make cproto flex bison gcc perl sed tr
do
  type $exe || error "no such executable, please install: $exe"
done

echo "### downloading pips"
svn checkout $PIPS_SVN/bundles/trunks $prod || error "cannot checkout pips"
PIPS_ARCH=`$prod/pips/makes/arch.sh`
export PIPS_ARCH

echo "### downloading $POLYLIB"
cd /tmp
test -f $POLYLIB.tar.gz && error "some $POLYLIB.tar.gz file already there"
wget $POLYLIB_SITE/$POLYLIB.tar.gz || error "cannot wget polylib"

echo "### building $POLYLIB"
gunzip $POLYLIB.tar.gz || error "cannot decompress polylib"
tar xf $POLYLIB.tar || error "cannot untar polylib"
cd $POLYLIB || error "cannot cd into polylib"
./configure --prefix=$prod/extern || error "cannot configure polylib"
make || error "cannot make polylib"

mkdir $prod/extern || error "cannot mkdir $prod/extern"
make install || error "cannot install polylib"
cd .. || error "cannot cd .."
rm -rf $POLYLIB || error "cannot remove polylib"
rm -f $POLYLIB.tar || error "cannot remove polylib tar"

echo "### fixing $POLYLIB"
mkdir $prod/extern/lib/$PIPS_ARCH || error "cannot mkdir"
cd $prod/extern/lib/$PIPS_ARCH || error "cannot cd"
ln -s ../libpolylib*.a libpolylib.a || error "cannot create links"

echo "### building newgen"
cd $prod/newgen
make compile

echo "### building linear"
cd $prod/linear
make compile

echo "### building pips"
cd $prod/pips
make compile

echo "### creating pipsrc.sh"
cat <<EOF > $destination/pipsrc.sh
# minimum rc file for sh-compatible shells

# default architecture
PIPS_ARCH=$PIPS_ARCH
export PIPS_ARCH

# subversion repositories
NEWGEN_SVN=$SVN_CRI/newgen
export NEWGEN_SVN

LINEAR_SVN=$SVN_CRI/linear
export LINEAR_SVN

PIPS_SVN=$SVN_CRI/pips
export PIPS_SVN

# software roots
EXTERN_ROOT=$prod/extern
export EXTERN_ROOT

NEWGEN_ROOT=$prod/newgen
export NEWGEN_ROOT

LINEAR_ROOT=$prod/linear
export LINEAR_ROOT

PIPS_ROOT=$prod/pips
export PIPS_ROOT

# path
PATH=\$PIPS_ROOT/bin:\$PIPS_ROOT/utils:\$PATH
EOF

# this should fail if not a developer
echo "### getting pips development"
svn checkout $PIPS_SVN/branches/$developer $destination/pips_dev
