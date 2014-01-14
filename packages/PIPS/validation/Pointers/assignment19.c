/* Do not assign undefined values
 *
 * Suggested by Pierre Jouvelot, but infirmed by the C standard
 *
 * The error only occurs when the pointer is dereferenced
 */

int assignment19() {
  int *p, *q;
 // the value of q is unknown, the assignment is meaningless, and gcc
  // issues a warning, but this is OK with the C standard
  p = q;
  // But this dereferencing is not
  return *p;
}
