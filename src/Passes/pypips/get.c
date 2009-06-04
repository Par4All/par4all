/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
double get(double f[SIZE],int i)
{
    return f[i];
}
void foo(double A[SIZE], double B[SIZE][SIZE]);
int main(int argc, char **argv)
{
    int i,j;
    double a[SIZE],b[SIZE][SIZE];
    double s=0;
    for(i=0;i<SIZE;i++)
    {
        a[i]=rand();
        for(j=0;j<SIZE;j++)
            b[i][j]=rand();
    }
#ifdef TEST
#ifndef rdtscll
#define rdtscll(val) \
         __asm__ __volatile__("rdtsc" : "=A" (val))
#endif
    long long stop,start;
    rdtscll(start);
    foo(a,b);
    rdtscll(stop);
    printf("%lld\n",stop-start);
#else
    foo(a,b);
#endif
    for(i=0;i<SIZE;i++)
        for(j=0;j<SIZE;j++)
            s+=a[i]+b[i][j];
    return (int)s;
}

#ifdef TEST
int MAX0(int a,int b) { return ((a)>(b))?(a):(b); }
#endif

