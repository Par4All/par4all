/* Check that successive external floating point dependent initializationa are properly taken
   into account. */

void static01()
{
  static int i = 0.;

  i++;
  printf("%d\n", i);
}

main()
{
  static01();
  static01();
  static01();
}
