/* Check the dereferencing of an undefined pointer
 *
 * Should not be called null06.c but undefinedxx.c
 */

int main()
{
  int i, * p=&i, *q;

  // An error should be detected according to Pierre Jouvelot, because
  // the value of q is undefined and then p is assigned an undefined
  // value too. This is detected by gcc.
  p = q;

  return 0;
}
