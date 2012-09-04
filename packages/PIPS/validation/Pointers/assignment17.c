/* Chasing bugs related no non-strict pointer typing
 *
 * Simplified version of assignment12.c to find bugs related to qq
*/

void assignment17(int **qq) {
  int *p;
  p = *qq;
  return;
}


