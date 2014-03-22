#!/bin/bash
#
# $Id$
#
# Compile PIPS from scratch in a temporary directory.
# Can be run from cron.
#
# $0 name log [email]

name=$1 log=$2 email=$3

url="https://scm.cri.ensmp.fr/svn/nlpmake/trunk/makes/setup_pips.sh"
dir="/tmp/pips_compile_from_scratch.$$"
set="./setup.sh"

function report()
{
  local status=$1 message=$2
  echo "$name: $message" >&2
  {
      echo "script: $0"
      echo "name: $name"
      echo "dir: $dir"
      echo "duration: ${SECONDS}s"
      echo "status: $status"
      echo "message: $message"

      if [ $status != 0 ] ; then
        echo
        echo "### OUT"
        test -f out && tail -100 out
        echo
        echo "### REPORT"
        test -f err && tail -100 err
      fi
  } > $log
  if [ "$email" ] ; then
    mail -s "$name: $message" $email < $log
  else
    [ $status -ne 0 ] && cat $log
  fi
  exit $status
}

# various check & setup
test $# -ge 2 > out 2> err || report 1 "usage: $0 name log [email]"
type curl >> out 2>> err || report 2 "curl not found"
mkdir $dir >> out 2>> err || report 3 "cannot create $dir"
cd $dir >> out 2>> err || report 4 "cannot cd to $dir"
curl -s -o $set $url >> out 2>> err || report 5 "cannot get $url"
chmod a+rx $set >> out 2>> err || report 6 "cannot chmod $set"
type timeout >> out 2>> err || report 7 "no timeout command"

# must compile pips under 20 minutes
timeout 20m $set PIPS calvin export < /dev/null > out 2> err || \
  report 8 "error running $set"

# checks
ROOT=$dir/PIPS/prod/pips
arch=$ROOT/makes/arch.sh
test -x $arch >> out 2>> err || report 10 "no arch script: $arch"
tpips=$ROOT/bin/$($arch)/tpips
test -x $tpips >> out 2>> err || report 11 "no generated tpips ($tpips)"

# run something!
source ./PIPS/pipsrc.sh >> out 2>> err || \
  report 9 "cannot source pips environment"
cat > foo.c 2>> err <<EOF
int main(void) {
  int i = 3;
  i -= 3;
  return 0;
}
EOF

$tpips >> out 2>> err <<EOF || report 12 "cannot run tpips ($tpips)"
create foo foo.c
activate PRINT_CODE_PRECONDITIONS
display PRINTED_FILE
close
delete foo
EOF

grep '{i==0}' out > /dev/null 2>> err || report 13 "precondition not found"

# cleanup
cd $HOME
rm -rf $dir >> out 2>> err || report 14 "cannot remove directory"

# done
report 0 "pips scratch compilation ok"
