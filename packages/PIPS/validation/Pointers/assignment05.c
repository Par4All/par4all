/* Check the generation of a target for a formal parameter
 *
 * FI: q is not properly assigned -> bug
 *
 * FI: I do not understand by p points towards _p_1[0] instead of _p_1
 * This is linked to property POINTS_TO_STRICT_POINTER_TYPES
 */

void assignment05(int *p)
{
  int i;
  int * q;

  q = p;
  i++; // FI: i++ added to avoid a possible issue with "return"
  return;
}
