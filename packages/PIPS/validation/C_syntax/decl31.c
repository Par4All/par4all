/* Check the type declaration when a formal parameter's type is
   parametric with respect to another formal parameter. We want the
   second occurence of n to be linked to the same entity as the first
   one. */

void decl31(int n, double a[n])
{
  a[0] = 0.;
}
