#! /bin/bash
#
# $Id$
#
# basic compile - run - compare validation script
# can be called from any validation script with one argument
#
# useful environment variables:
# - PIPS_VALIDATION_EXE: triggers compiling & running
# - PIPS_CC: C compiler
# - PIPS_F77: Fortran 77 compiler
# - PIPS_F90: Fortran 90 compiler
# - PIPS_DIFF: diff command
#

# err <exit-status> <error-message>
function err()
{
  local status=$1 msg="$2"
  echo "## $0 $msg"
  echo "## $0 $msg" >& 2
  exit $status
}

# defaults
initial=1
generated=1
compile=1
run=1
compare=1
message="compile run compare"
what="both"
copt=
file=
exit=0
display=

# option management
# some more stuff could be added here if needed
while [[ $1 == -* ]]
do
  opt=$1
  shift
  case $opt in
    # end of options
    --) break ;;
    # help!
    -h|--help)
      echo -e \
	  "usage: $0 [options] dbname\n" \
	  "dbname is the prefix of the pips database, without the .database\n" \
          "options include:\n" \
	  "  -h: this help\n" \
	  "  -i: process initial version\n" \
	  "  -g: process generated version\n" \
	  "  -b: process both initial & generated versions\n" \
	  "  -c: compile only\n" \
	  "  -cr: compile and run\n" \
	  "  -crc: compile, run and compare\n" \
          "  -e: expect this exit status from the program (default is 0)\n" \
	  "  -f file: use this file instead of dbname.[cf...]\n" \
	  "  -d: display output (not really portable across compilers)\n"
      exit 0;
      ;;
    # initial or generated only
    -i|--i|--initial)
      initial=1 generated=
      what="initial"
      ;;
    -g|--g|--generated)
      initial= generated=1
      what="generated"
      ;;
    -b|--b|--both)
      initial=1 generated=1
      what="both"
      ;;
    # operations to perform
    -c|--c|--compile)
      compile=1 run= compare= initial= copt=-c
      message="compile"
      ;;
    -cr|--cr|--compile-run)
      compile=1 run=1 compare= initial=
      message="compile run"
      ;;
    -crc|--crc|--compile-run-compare)
      compile=1 run=1 compare=1 initial=
      message="compile run compare"
      ;;
    # display: do not use, this more for debug...
    # the display cannot be performed when PIPS_VALIDATION_EXE is off
    -d|--display) display=1
      ;;
    # expected exit status
    -e|--exit) exit=$1 ; shift
      ;;
    --exit=*) exit=${opt//*=/}
      ;;
    # file name, if not case.suffix
    -f|--file) file=$(basename $1) ; shift
      ;;
    --file=*) file=$(basename ${opt//*=/})
      ;;
    # option error
    *) err 14 "unexpected option $opt" ;;
  esac
done

if [ "$file" ]
then
  [ -e $file ] || err 15 "expected file $file not found"
fi

# argument management
if [ $# -ne 1 ]
then
  err 1 "expecting 1 argument, got $# ($@)"
else
  case=$1
  shift
fi

# this is the only expected output, whether the case is run or not
echo -e "#\n# $message $what $case\n#"

# get suffix & set compiler accordingly
if [[ -e $case.c || "$file" == *.c ]]
then
  suffix=c
  comp=${PIPS_CC:-cc}
elif [[ -e $case.f || "$file" == *.f ]]
then
  suffix=f
  comp=${PIPS_F77:-gfortran}
elif [[ -e $case.f90 || "$file" == *.f90 ]]
then
  suffix=f90
  comp=${PIPS_F90:-gfortran}
else
  err 2 "no source found for case $case"
fi

# overall settings
source=${file:-$case.$suffix}
database=$case.database
tmp=$database/Tmp
unsplit=$database/Src/$source

# common prefix for all temporary files
exe=$tmp/exe

# some sanity checks. the first one cannot fail.
[ "$initial" ] && {
  [ -e $source ] || err 9 "expected source $source not found"
}
[ "$generated" ] && {
  [ -d $database ] || err 10 "expected database $database not found"
  [ -e $unsplit ] || err 11 "expected unsplit $unsplit not found"
}

# check trigger
[ "$PIPS_VALIDATION_EXE" ] || exit 0

# ok, we are doing the stuff...
echo -e "### compiling, running & comparing $case..." >&2

# check for the compiler
type $comp >&2 || err 8 "compiler $comp not found"

# result are stored in a temporary directory within the database
[ -d $tmp ] || mkdir $tmp || err 12 "cannot create temporary directory $tmp"

# compile both initial & unsplit source
if [ "$compile" ]
then
  if [ "$initial" ] ; then
    $comp $copt -o $exe.1 $source ||
      err 3 "cannot compile initial source $source"
  fi
  if [ "$generated" ] ; then
    $comp $copt -o $exe.2 $unsplit ||
      err 4 "cannot compile unsplit source $unsplit"
  fi
fi

# run both exes
if [ "$run" ]
then
  if [ "$initial" ] ; then
    $exe.1 > $exe.1.out 2> $exe.1.err
    status=$?
    [ $status -eq $exit ] || \
      err 5 "status $status instead of $exit while executing initial code"
    if [ "$display" ] ; then
      echo "# initial version output"
      cat $exe.1.out
    fi
  fi
  if [ "$generated" ] ; then
    $exe.2 > $exe.2.out 2> $exe.2.err
    status=$?
    [ $status -eq $exit ] || \
      err 6 "status $status instead of $exit while executing unsplit code"
    if [ "$display" ] ; then
      echo "# generated version output"
      cat $exe.2.out
    fi
  fi
fi

# compare result output
if [ "$compare" -a "$initial" -a "$generated" ]
then
  ${PIPS_DIFF:-diff} $exe.1.out $exe.2.out > $exe.diff ||
    err 13 "cannot run diff command"

  # now check for any difference, and report
  if [ -s $exe.diff ]
  then
    # not good, this is a fail
    #echo "some differences on $case..." >&2
    cat $exe.diff
    err 7 "comparison yields to differences..."
  fi
else
  # when running & not comparing, should show some results?
  # hmmm... it is not portable
  :
fi

# all is well, cleanup
rm -f $exe.*
exit 0
