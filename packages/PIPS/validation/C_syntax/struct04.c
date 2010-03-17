/* Make sure that the struct s is fully expanded in the declaration of
   the anonymous union because it has not been declared earlier */

union
{
  struct s
  {
    int l;
  } d;
  int i;
} u;
