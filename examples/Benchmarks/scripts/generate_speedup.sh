#!/bin/bash

if [[ -z $dbfile ]]; then
dbfile="timing.sqlite"
fi
echo $dbfile

out_dat="timing.dat"
out_gp="histogram.gp"
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
tests=`echo "select testcase from timing $exclude_tests group by testcase;" | sqlite3 $dbfile`
fi

# 1st line is header
echo -n " run_seq" >> $out_dat
nvers=0 #compute number of versions
for ver in $versions; do
  nvers=$(($nvers + 1))
  ver=`echo $ver|sed 's/run_//g'|sed 's/_/ /g'`
  echo -n " \"$ver\"" >> $out_dat
done
echo >> $out_dat


mean_idx=0
for ver in $versions; do
  mean_expr[$mean_idx]="0" # compute geomean !
  ((mean_idx++))
done

nmean=0

for test in $tests; do
  echo -n $test >> $out_dat
  # reference 
  ref=`echo "select ROUND(AVG(measure),2) from timing where testcase=\"$test\" and version=\"$ref_ver\";" | sqlite3 $dbfile`
  mean_idx=0
  for ver in $versions; do
    time=`echo "select ROUND(AVG(measure),2) from timing where testcase=\"$test\" and version=\"$ver\";" | sqlite3 $dbfile`
    speedup=0
    if [[ ! -z $time && ! -z $ref ]]; then
      #echo "speedup $test $ver $time : $ref/$time"
      speedup=`echo "scale=2; $ref/$time" | bc `
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
done

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



cp speedup.gp $out_gp

echo -n "plot " >> $out_gp
nver=2
ratio=`echo "scale=2; 1/$nvers" | bc `
labelfontsize=$((11-${nmean}/4))
for ver in $versions; do
  echo "\\" >> $out_gp
  echo "$sep '$out_dat' u $nver:xtic(1) \\" >> $out_gp
  echo -n ",'' u (\$0-(${nvers}-2)/2.*1./(${nvers}+2)+($(($nver-2)))*1./(${nvers}+g)):$nver:$nver w labels font 'Arial,$labelfontsize' left rotate by 90 offset 0,0.1 t ''" >> $out_gp
  nver=$(($nver+1))
  sep=","
done
echo >> $out_gp
