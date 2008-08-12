
gcc -O6 -mmmx -msse $1.database/MAIN/MAIN.c $1.database/$1REF/$1REF.c $1.database/$1/$1.c -I . -o $1.out

#gcc -O6 -mmmx -msse $1.database/$1/$1.c -I . -S

#rm -r -f $1.database