/* example from Serge Guelton */
/* lots of array linearization */

#include <stdio.h>

int duck1(int riri[10], int fifi[2][3], int size, int loulou[1][size][6])
{
   int *zaza = (int *) fifi+(3-1-0+1)*1;
   /* proper effects are not precise here for loulou because 
      of the internal representation of the expression */
   return *((int *) riri+2) = *(zaza+1)+*((int *) loulou+3+(6-1-0+1)*(0+(size-1-0+1)*0));
}

int duck2(int riri[10], int fifi[2][3], int size, int loulou[1][size][6])
{
   int *zaza = (int *) fifi+(3-1-0+1)*1;
   return *((int *) riri+2) = *(zaza+1)+*((int *) loulou+(3+(6-1-0+1)*(0+(size-1-0+1)*0)));
}

int main()
{
   int riri[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
   int fifi[2][3] = {{10, 11, 12}, {13, 14, 15}};
   int size = 2;
   int loulou[1][size][6];
   int i;
   int j;
   int k = 16;
   for (i = 0;i<size;i++)
      for (j = 0;j<6;j++)
         loulou[0][i][j] = k++;
   printf("%d\n", duck(riri, fifi, size, loulou));
   return 0;
}
