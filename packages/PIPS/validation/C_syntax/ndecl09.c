/* Purpose: what's wrong with struct? */

void ndecl09(char *dir)
{
  /* Well, x is a struct of name m: x declaration disappears... */
  struct m {
    int a;
    float b;
  } x, *px, foo(), (*bar)();

}
