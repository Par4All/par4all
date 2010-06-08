#include <stdio.h>

int partial_eval02(int riri[10], int fifi[2][3], int size, int loulou[1][size][6])
{
  int *zaza = (int *) fifi+(3-1-0+1)*1;
  int i;

  i = size;
  return *((int *) riri+2) = *(zaza+1)+*( &loulou[0][0][0]+3+(6-1-0+1)*(0+(size-1-0+1)*0));
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

  i = size;

  for (i = 0;i<size;i++)
    for (j = 0;j<6;j++)
      loulou[0][i][j] = k++;

  printf("%d\n", partial_eval02(riri, fifi, size, loulou));

  return 0;
}
