#! /bin/sh
#
# $Id$
#
# Setup a pips installation from scratch
#

# where to get pips
PIPS_SVN='http://svn.cri.ensmp.fr/svn/pips'

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

echo "### downloading pips"
svn checkout $PIPS_SVN/bundles/trunks $prod || error "cannot checkout pips"
PIPS_ARCH=`$prod/pips/makes/arch.sh`
export PIPS_ARCH

echo "### downloading $POLYLIB"
cd /tmp
test -f $POLYLIB.tar.gz && error "some $POLYLIB.tar.gz file already there"
wget $POLYLIB_SITE/$POLYLIB.tar.gz || error "cannot wget polylib"

echo "### building $POLYLIB"
tar xzf $POLYLIB.tar.gz || error "cannot untar polylib"
cd $POLYLIB || error "cannot cd into polylib"
./configure --prefix=$prod/extern || error "cannot configure polylib"
make || error "cannot make polylib"

mkdir $prod/extern || error "cannot mkdir $prod/extern"
make install || error "cannot install polylib"
cd .. || error "cannot cd .."
rm -rf $POLYLIB || error "cannot remove polylib"
rm -f $POLYLIB.tar.gz || error "cannot remove polylib tar"

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

PIPS_ARCH=$PIPS_ARCH
export PIPS_ARCH

EXTERN_ROOT=$prod/extern
export EXTERN_ROOT

NEWGEN_ROOT=$prod/newgen
export NEWGEN_ROOT

LINEAR_ROOT=$prod/linear
export LINEAR_ROOT

PIPS_ROOT=$prod/pips
export PIPS_ROOT

PATH=\$PIPS_ROOT/bin:\$PIPS_ROOT/utils:\$PATH
EOF

echo "### getting pips development"
svn checkout $PIPS_SVN/branches/$developer $destination/pips_dev
