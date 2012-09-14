// array of pointers towards arrays

#include <stdio.h>

void pointer15(double * (*t[3][4])[5][6][7])
{
  double z;
  (*(t[1][2]))[3][4][5] = &z;
  *(*(t[1][2]))[3][4][5] = 2.5;
  return;
}

int main()
{
  double * d[3][4][5][6][7];
  double * (*pd[3][4])[5][6][7];
  int i, j;
  for(i=0;i<3;i++)
    for(j=0;j<4;j++)
      pd[i][j] = &d[i][j];
  pointer15(pd);
  printf("%f\n", *d[1][2][3][4][5]);
  return 0;
}
