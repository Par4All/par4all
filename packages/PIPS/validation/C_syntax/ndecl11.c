/* Purpose: what's wrong with enum? */

void ndecl11(char *dir)
{
  /* Well, x is a enum of name m: x declaration disappears... */
  enum m {A, B, C  } x, *px, foo();
}
