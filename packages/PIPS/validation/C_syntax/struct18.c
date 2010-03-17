/* Make sure that the struct s is fully expansed in the declaration of
   the anonymous union because it has not been declared earlier.

   Equivalent to struct04.c, but within a function definition. So it's
   not parsed as a compilation unit.
 */

void struct18()
{
  union
  {
    struct s
    {
      int l;
    } d;
    int i;
  } u;
}
