/* Chasing bugs related no non-strict pointer typing in the presence
 * of pointer arithmetic.
 *
 * Same as assignment 12.c, but two increment added to make sure that
 * key points-to information is not lost.
 *
 * Since pp and qq are passed by value. the pointer arithmetic has no
 * impact on the out points-to set. We need another example with
 * (*pp)++ and/or (*qq)++ to have new interprocedural effects.
 */

void assignment16(int **pp, int **qq) {
  *pp = *qq;
  pp++;
  qq++;
  (*pp)++;
  //(*qq)++;
  return;
}


