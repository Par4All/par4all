/* Check that successive external dependent initializationa with side-effects are properly taken
   into account: forbidden by gcc. */

int delta = 1;
int delta2 = ++delta+2;

main()
{
  int i = 0;

  i = i + delta;
  i = i + delta2;
}
