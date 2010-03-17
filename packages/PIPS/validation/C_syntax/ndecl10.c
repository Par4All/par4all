/* Purpose: what's wrong with struct? */

void ndecl10(char *dir)
{
  /* Well, x is a union of name m: x declaration disappears... */
  union m {
    int a;
    float b;
  } x, *px, foo();

}
