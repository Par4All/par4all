/* Check recursive stub generation */

int formal_parameter03(int ppp[10][10][10])
{
  int *q = ppp[0][0];
  int (*qq)[10] = ppp[0];
  int (*qqq)[10][10] = ppp;

  return 0;
}
