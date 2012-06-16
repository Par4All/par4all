/* Check indexing of implicit arrays
 */

int properties04(int **q)
{
  int i;

  *q = &i;

  return 0;
}
