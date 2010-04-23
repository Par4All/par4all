/* Bug: z does not seem to have a pointer type. */

p3(x, z)
double x, *z;
{
  *z = x/2;
}
