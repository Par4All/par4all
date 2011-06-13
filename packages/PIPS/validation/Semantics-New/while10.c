/* Make sure the side effect of a non-entered loop is taken into
 * account
 *
 * Note that we run into trouble with C semantics by writing the
 * condition j<=10 && i++...
 */

int main()
{
  int c = 0;
  int i = 0;
  int j = 0;

  while(i++ && j<=10) {
    ;
  }
  return;
}
