/* Do not assign undefined values
 *
 * Suggested by Pierre Jouvelot
*/

void assignment19() {
  int *p, *q;
  p = q; // the value of q is unknown, the assignment is meaningless
  return;
}


