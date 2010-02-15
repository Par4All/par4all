/* Make sure that the struct s is fully expansed in the declaration of
   the anonymous union because it has not been declared earlier

   Same as struct04.c, but with two struct nested
 */

struct
{
  struct s
  {
    int l;
  } d;
  int i;
} u;
