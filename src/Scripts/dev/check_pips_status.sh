#! /bin/bash
#
# $Id$
#
# check pips compilation & validation status for nagios
# this scripts assumes a setup with pips_check_compile
#
# usage: $0 warn crit /path/to/prod [/path/to/validation...]

# for nagios?
PATH=/usr/lib/nagios/plugins:$PATH

function res()
{
  local status=$1 msg=$2
  case $status in
    0) echo "Pips OK: $msg";;
    1) echo "Pips WARNING: $msg";;
    2) echo "Pips CRITICAL: $msg";;
    3) echo "Pips UNKNOWN: $msg";;
    *) echo "Pips ERROR: $msg";;
  esac
  exit $status
}

[ $# -ge 3 ] || res 3 "expecting at least 3 args"

warn=$1 crit=$2 prod=$3
shift 3

name=$(basename $(dirname $prod))

# compilation status
cd $prod || res 3 "$name no prod directory '$prod'"
check_file_age -w $warn -c $crit .svn > /dev/null || \
  res $? "$name svn update"
check_file_age -w $warn -c $crit CURRENT > /dev/null || \
  res $? "$name CURRENT file"

test -f STATE || res 3 "$name no STATE file"
read current compile < STATE

case $current in
  ok) ;;
  KO:*)   res 2 "$name state is $current $compile" ;;
  locked) res 3 "$name is locked";;
  *)      res 4 "$name in unexpected state $state $compile" ;;
esac

# validation status
for valid in "$@"
do
  [ -d $valid ] || res 3 "$name no validation directory '$valid'"
  cd $valid || res 2 "$name cd '$valid'"
  check_file_age -w $warn -c $crit .svn > /dev/null || \
    res $? "$name svn update '$valid'"
  # check_file_age -w $warn -c $crit SUMMARY.short > /dev/null
  [ -f SUMMARY.short ] || res 2 "$name no SUMMARY.short in '$valid'"
  status=$(tail -1 SUMMARY.short)
  read n a v what remain <<EOF
$status
EOF
  case $what in
    SUCCESS) ;;
    ISSUES) res 1 "$name validation failed '$valid'" ;;
    *)      res 3 "$name unexpected status $what in $valid" ;;
  esac
done

res 0 "$name is fine"
