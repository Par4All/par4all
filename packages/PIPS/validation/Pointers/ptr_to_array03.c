/* Pointer to array
 *
 * Same as ptr_to_array01.c, but ptr_to_array03.tpips uses property
 * POINTS_TO_NULL_POINTER_INITIALIZATION to obtain p->_p_1 EXACT.
 */

int ptr_to_array03(int (*p)[10])
{
  (*p)[3] = 1;

  return 0;
}
