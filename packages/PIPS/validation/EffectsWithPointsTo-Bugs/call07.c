/* #include<stdio.h> */

void call07(int r, int m, int *x)
{
  int j;

      x[4] = 10*r+4;
}

main()
{
  int d1 = 4;
  int d2 = 4;
  int y[d1][d2];
  int i, j;

  for(i=0;i<d1;i++)
    call07(i, d2, &(y[i][0]));

  /*
  for(i=0;i<d1;i++) {
    for(j=0;j<d2;j++)
      printf("y[%d][%d] = %d\t", i, j, y[i][j]);
    printf("\n");
  }
  */
}
