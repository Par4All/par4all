#!/bin/bash


out_dat="timing.dat"
out_gp="histogram.gp"

#HEADER
cat > $out_gp << EOF
reset

set xrange[-0.5:25]
set yrange[-2:9]

set ytics(              \
	"0.25x"  -2,    \
	"0.5x"   -1,    \
	"1x"      0,    \
	"2x"      1,    \
	"4x"      2,    \
	"8x"      3,    \
	"16x"     4,    \
	"32x"     5,    \
	"64x"     6,    \
        "128x"    7,    \
        "256x"    8)

#set x2tics("PolyBench" -0.5, "Rodinia" 19.5, "Other" 22.5, "" 23.5)

# We need 'in' ticks, so that the suite labels are
# closer to the boundary
set x2tics in nomirror offset 5
# We then use arrows to add-back the out-labels
set arrow 6 from -0.5,9 to -0.5,10 nohead
set arrow 7 from 19.5,9 to 19.5,10 nohead
set arrow 8 from 22.5,9 to 22.5,10 nohead
set arrow 9 from 23.5,9 to 23.5,10 nohead

set xtics out nomirror
set xtics rotate by -90

# # divides Polybench from Rodinia
# # divides Rodinia from others
# # Divides Otehrs from geomean
# set arrow 1 from 19.6125,-2 to 19.6125,8 nohead
# set arrow 2 from 22.6125,-2 to 22.6125,4.25 nohead
# set arrow 3 from 22.6125,8 to 22.6125,7.5 nohead
# set arrow 4 from 23.6125,-2 to 23.6125,4.25 nohead
# set arrow 5 from 23.6125,8 to 23.6125,7.5 nohead


set grid ytics noxtics x2tics

g = 1
set style histogram cluster gap 1
set style data histogram

#set key right width -3

unset xtics
set xtics out rotate by -30 nomirror offset 0,0 font "Arial,12" 
#set key outside center bottom horizontal Right noreverse enhanced autotitles columnhead nobox 
set key Right noreverse enhanced autotitles columnhead nobox 

set style fill solid border -1

set size 1, 0.5

set ylabel "Speedup (Log_{2})"

set terminal postscript eps enhanced "Times-New-Roman" 10 size 9,4
set output "speedup.eps"

EOF



if [[ -z $dbfile ]]; then
dbfile="timing.sqlite"
fi
echo $dbfile

rm -f $out_dat;

if [[ ! -e $dbfile ]]; then
  echo "There's no sqlite db to process (${dbfile})"
fi

if [[ -z $versions ]]; then
versions=`echo "select version from timing group by version;" | sqlite3 $dbfile`
fi
if [[ -z $ref_ver ]]; then
ref_ver=run_seq
fi

if [[ ! -z $exclude_tests ]]; then
exclude_tests="where testcase not in ($exclude_tests)"
fi

if [[ -z $tests ]]; then
tests=`echo "select testcase from timing $exclude_tests group by testcase order by suite_order,testcase;" | sqlite3 $dbfile`
fi


if [[ -z $min_speedup ]]; then
min_speedup=0.1
fi

if [[ -z $strip_ver ]]; then
strip_ver=run_
fi

# 1st line is header
echo -n " run_seq" >> $out_dat
nvers=0 #compute number of versions
for ver in $versions; do
  nvers=$(($nvers + 1))
  ver=`echo $ver|sed "s/$strip_ver//g"|sed 's/_/ /g'`
  echo -n " \"$ver\"" >> $out_dat
done
echo >> $out_dat


mean_idx=0
for ver in $versions; do
  mean_expr[$mean_idx]="0" # compute geomean !
  ((mean_idx++))
done

nmean=0

currentSuite="unset"
currentTic=-0
for test in $tests; do
  suite=`echo "select bench_suite from timing where testcase=\"$test\" limit 1;" | sqlite3 $dbfile`
  if [ "$suite" != "$currentSuite" ]; then
    tic=`echo "$currentTic -0.5" | bc `
    x2tics="$x2tics $x2sep \"$suite\" $tic"
    x2sep=","
    currentSuite=$suite
  fi
      

  echo -n $test >> $out_dat
  # reference 
  ref=`echo "select ROUND(AVG(measure),2) from timing where testcase=\"$test\" and version=\"$ref_ver\";" | sqlite3 $dbfile`
  mean_idx=0
  for ver in $versions; do
    time=`echo "select ROUND(AVG(measure),2) from timing where testcase=\"$test\" and version=\"$ver\";" | sqlite3 $dbfile`
    speedup=0
    if [[ ! -z $time && ! -z $ref ]]; then
      #echo "speedup $test $ver $time : $ref/$time"
      speedup=`echo "scale=1; $ref/$time" | bc `
      assert_min=`echo "$speedup < $min_speedup" | bc`
      if [[ $assert_min == 1 ]]; then
        echo "$speedup < $min_speedup" $assert_min
        speedup=$min_speedup
      echo "$test $ver $speedup"
      fi
    fi

    if [[ "$speedup" ==  "0" ]]; then
      mean_expr[$mean_idx]+="+ 0"
    else 
      mean_expr[$mean_idx]+="+ l($speedup)"
    fi
    ((mean_idx++))
    echo -n " $speedup" >> $out_dat
  done
  echo >> $out_dat
  nmean=$((nmean+1))
  currentTic=$(($currentTic+1))
done

tic=`echo "$currentTic -0.5" | bc `
x2tics="$x2tics $x2sep \"$suite\" $tic"
echo "set x2tics($x2tics)" >> $out_gp

if [[ -z $disable_mean ]]; then
  element_count=${#mean_expr[@]}
  mean_idx=0
  echo -n "Geo.Mean" >> $out_dat
  while [ "$mean_idx" -lt "$element_count" ]
  do    # List all the elements in the array.
    echo "e((${mean_expr[$mean_idx]})/$nmean)"
    geomean=`echo "scale=2; e((${mean_expr[$mean_idx]})/$nmean)" | bc -l`
    echo "$geomean"
    echo -n " $geomean" >> $out_dat
    ((mean_idx++))
  done
  echo >> $out_dat
fi



if [[ ! -z $title ]]; then
echo set title '"'$title'"' >> $out_gp
fi


echo -n "plot " >> $out_gp
nver=2
if [[ -z $labelfontsize ]] ; then
  labelfontsize=$((11-${nmean}/4))
fi
for ver in $versions; do
  echo "\\" >> $out_gp
  color=`echo "scale=2; e($nver-2)/e($nvers-1)" | bc -l`
  echo "$sep '$out_dat' u (log(\$$nver)/log(2)):xtic(1) lt 1 fs solid $color \\" >> $out_gp
  echo -n ",'' u (\$0-(${nvers}-g)/2.*1./(${nvers}+2)+($(($nver-2)))*1./(${nvers}+g)):(log(\$$nver)/log(2)):(log(\$$nver)/log(2)) w labels font 'Arial,$labelfontsize' right rotate by -90 offset -0.3,0.1 t ''" >> $out_gp
  nver=$(($nver+1))
  sep=","
done
echo >> $out_gp
