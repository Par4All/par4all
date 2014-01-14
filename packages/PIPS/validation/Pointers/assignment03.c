/* Check a pointer copy */

int assignment03()
{
  int i;
  int * p;
  int * q;

  p = &i;
  // Copy of an undefined/indeterminate pointer allowed by C standard
  p = q;

  // Not OK with C standard; this shows in points-to OUT
  return *p;
}
