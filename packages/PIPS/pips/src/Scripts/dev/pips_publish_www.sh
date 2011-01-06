#! /bin/bash
#
# $Id$
#
# publish pips documentation at CRI

# some parameters
rsync='rsync -ravC --delete'
ddir=/projects/publics_html

# check
test -d $ddir || { echo "no such directory: $ddir" ; exit 1 ; }

function fixperms()
{
  find $1 -type f -print0 | xargs -0 chmod a+r
  find $1 -type d -print0 | xargs -0 chmod a+rx
}

for d in newgen pips ; do
  src=$d/doc
  dst=$ddir/$d
  test -d $src || { echo "no such doc dir: $src" ; exit 2 ; }
  test -d $dst || { echo "no such dst doc dir: $dst" ; exit 3 ; }
  $rsync $src/. $dst/.
done

# cleanup doxygen
test -d doxygen && rm -rf doxygen

# do not use current make file doxygen targets
# the same files are generated several times
{
  cat pips/makes/share/doxygen/Doxyfile
  echo "INPUT = newgen/src/genC linear/src pips/src/Libs pips/src/Passes pips/src/Documentation/newgen"
  # let us skip gcc sources & .svn dirs
  echo "EXCLUDE_PATTERNS = */.svn/* */Passes/fortran95/gcc* */Passes/fortran95/build/*"
  echo "PROJECT_NAME = PIPS"
  echo "OUTPUT_DIRECTORY = doxygen"
  echo "GENERATE_LATEX = NO"
  echo "HAVE_DOT = YES"
  # show directories & path
  echo "SHOW_DIRECTORIES = YES"
  echo "FULL_PATH_NAMES = YES"
  echo "STRIP_FROM_PATH ="
} | doxygen - > doxy.out 2> doxy.err

$rsync doxygen/html/. $ddir/doxygen/.

# fix permissions
fixperms $ddir/newgen
fixperms $ddir/pips
fixperms $ddir/doxygen
