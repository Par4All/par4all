/* Check that successive external dependent initializationa are properly taken
   into account.

   This is not standard C.
 */

int delta = 1;
int delta2 = delta+2;

main()
{
  int i = 0;

  i = i + delta;
  i = i + delta2;
}
