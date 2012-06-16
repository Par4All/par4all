/* Chasing bugs related no non-strict pointer typing
*/

void assignment12(int **pp, int **qq) {
  *pp = *qq;
  return;
}


