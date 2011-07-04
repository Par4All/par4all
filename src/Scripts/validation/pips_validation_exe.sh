#! /bin/bash
#
# $Id$
#
# basic compile - run - compare validation script
# can be called from a test or tpips script with one argument
# useful environment variables:
# - PIPS_VALIDATION_EXE: triggers compiling & running
# - PIPS_CC: C compiler
# - PIPS_F77: Fortran 77 compiler
# - PIPS_F90: Fortran 90 compiler
# - PIPS_DIFF: diff command
#

if [ $# -ne 1 ]
then
  echo "$0 expecting 1 argument" >&2
  exit 1
else
  case=$1
  shift
fi

# this is the only expected output, whether the case is run or not
echo -e "#\n# compile run compare $case\n#"

# check trigger
[ "$PIPS_VALIDATION_EXE" ] || exit 0

# we are doing the stuff...
#echo -e "### compiling, running & comparing $case..." >&2

# get suffix & set compiler
if [ -e $case.c ]
then
  suffix=c
  comp=${PIPS_CC:-cc}
elif [ -e $case.f ]
then
  suffix=f
  comp=${PIPS_F77:-gfortran}
elif [ -e $case.f90 ]
then
  suffix=f90
  comp=${PIPS_F90:-gfortran}
else
  echo "no source found for case $case" >&2
  exit 2
fi

# compile both initial & unsplit source
#echo "compiling $case..." >&2
$comp -o $case.exe.1 $case.$suffix || exit 3
$comp -o $case.exe.2 $case.database/Src/$case.$suffix || exit 4

# run both exes
#echo "running $case..." >&2
./$case.exe.1 > $case.exe.1.out 2> $case.exe.1.err || exit 5
./$case.exe.2 > $case.exe.2.out 2> $case.exe.2.err || exit 6

# compare result output
#echo "comparing $case..." >&2
${PIPS_DIFF:-diff} $case.exe.1.out $case.exe.2.out > $case.exe.diff

if [ -s $case.exe.diff ]
then
  # not good, this is a fail
  #echo "some differences on $case..." >&2
  cat $case.exe.diff
  exit 7
else
  #echo "all is well for $case..." >&2
  # all is well, cleanup
  rm -f $case.exe.*
  exit 0
fi
