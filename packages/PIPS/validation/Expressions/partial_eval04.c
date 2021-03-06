/* Derived from partial_eval01.c to check analysis of static counters
   and global counters

   Pointer analysis must be activated because of zaza. Offsets in
   array must leace the pointers in the arrays. To check it, the
   return expression is broken down into three pieces.
 */

#include <stdio.h>

int duck_counter = 0;

int duck(int riri[10], int fifi[2][3], int size, int loulou[1][size][6])
{
  static int internal_duck_counter = 0;
  int *zaza = (int *) fifi+(3-1-0+1)*1;
  int i, j;
  internal_duck_counter++;
  printf("internal duck counter=%d\n", internal_duck_counter);
  duck_counter++;
  // return *((int *) riri+2) = *(zaza+1)+*((int *) loulou+3+(6-1-0+1)*(0+(size-1-0+1)*0));
  i = *(zaza+1);
  j = *((int *) loulou+3+(6-1-0+1)*(0+(size-1-0+1)*0));
  return *((int *) riri+2) = i+j;
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
  int t;
  for (i = 0;i<size;i++)
    for (j = 0;j<6;j++)
      loulou[0][i][j] = k++;
  t = duck(riri, fifi, size, loulou);
  printf("global duck counter=%d\n", duck_counter);
  printf("%d\n", t);
  return 0;
}
