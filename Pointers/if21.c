/* Francois Irigoin: see how intersection is handled at the CP level
 * and at the PT level.
 *
 * Designed to think about Section 5.3.5 in Chapter 5, intersection
 * operator on Vref.
 *
 * In fact, the intersection operator in the PhD is implemented as a
 * conflict detection operator, references_may/must_conflict_() in
 * effects-util/conflicts.c.
 *
 * The result is not as precise as it could/should be. The last update
 * of array "pp", "pp[l]=&pl;" does not invalidate the exactitude of
 * the arc "pp[1]->pl" because the target is the same in both cases.
 */

int main() {
  int i, j, k, l;
  int **pp[10], *pi=&i, *pj=&j, *pk=&k, *pl=&l;
  double z;

  if(z>0.)
    pp[i] = &pi;
  else
    pp[j] = &pj;

  pp[0] = &pk;

  pp[1] = &pl;

  pp[l] = &pl;

  return 0;
}
