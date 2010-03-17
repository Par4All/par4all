/* Proper handling of void functions */

int j = 0;

double call11(void)
{
  double x = 3.;
  j++;

  return x;
}

main()
{
  int ai = 3;

  call11();

  ai = 0;
}
