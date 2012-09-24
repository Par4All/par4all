/* Derived from dereferencin08, example from Serge Guelton
 *
 * Pointer "zizi" added.
 */

#include<stdio.h>

void bar(int (*a)[3])
{
  (*a)[0] = 998;
}

void foo(int *i)
{
  *i = 999;
}

void foobar(int size, int (*ap)[size][6])
{
  ap[0][0][0] = 1000;
}


int duck1(int riri[10], int fifi[2][3], int size, int loulou[20][size][6])
{
  // Here the array "fifi" is cast into an int *, and then an offset is computed
  // fifi+(3-1-0+1)*1 points towards fifi[0][3],
  int *zaza = (int *) fifi+(3-1-0+1)*1;

  // Here an offset is computed first
  // fifi+(3-1-0+1)*1 points towards fifi[3], and the cast makes it a
  // pointer towards fifi[3][0]
  int *zuzu = (int *) (fifi+(3-1-0+1)*1);

  // zizi points to fifi[3][0]
  int *zizi = ((fifi+(3-1-0+1)*1)[0]);

  printf("fifi=%p, zaza=%p, zizi=%p, zuzu=%p\n", fifi, zaza, zizi, zuzu);

  bar(fifi+(3-1-0+1)*1);
  foo((fifi+(3-1-0+1)*1)[0]);
  /* proper effects are not precise here for loulou because 
     of the internal representation of the expression */
  return *((int *) riri+2) = *(zaza+1)+*((int *) loulou+3+(6-1-0+1)*(0+(size-1-0+1)*0));
}

int duck2(int riri[10], int fifi[2][3], int size, int loulou[20][size][6])
{
   int *zaza = (int *) fifi+(3-1-0+1)*1;
   int *tmp = (int *) loulou+(3+(6-1-0+1)*(0+(size-1-0+1)*0));
   printf("tmp=%p, loulou=%p, tmp-loulou=%ld\n",tmp, loulou, (tmp-(int *)loulou)/4);
   foobar(size, loulou+(3+(6-1-0+1)*(0+(size-1-0+1)*0)));
   return *((int *) riri+2) = *(zaza+1)+*tmp;
}

int main()
{
   int riri[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
   int fifi[2][3] = {{10, 11, 12}, {13, 14, 15}};
   int size = 2;
   int loulou[20][size][6];
   int i;
   int j;
   int k = 16;
   int l;
   for(l=0; l<20; l++)
     for (i = 0;i<size;i++)
       for (j = 0;j<6;j++)
         loulou[l][i][j] = k++ + 100*l;
   printf("%d\n", duck1(riri, fifi, size, loulou));
   printf("%d\n", duck2(riri, fifi, size, loulou));
   return 0;
}
