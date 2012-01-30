/* To make sure casts are properly handled */

void lhs04()
{
  float x;

  *((int *) &x) &= 0x7fffffff;
}
