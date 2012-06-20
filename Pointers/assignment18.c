/* Chasing bugs related no non-strict pointer typing
 *
 * Simplified version of assignment12.c to find bugs related to qq
*/

void assignment18(int **qq) {
  int *p, i;
  p = *qq;
  i = *p;
  return;
}


