/* Check the dereferencing of an undefined pointer
 *
 * Should not be called null06.c but undefinedxx.c
 */

int main()
{
  int * p;

  *p = 1;

  return 0;
}
