/* To check pointer references
 *
 * Derived from pointer_reference02(), with p[3] used as a left-hand side
 */

int pointer_reference04(char **p)
{
  char * q = p[3];
  char * r = "hello";
  p[3] = r;
  return 0;
}
