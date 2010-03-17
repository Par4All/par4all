/* Purpose: see what happens with comments when a multiline
   declaration is reduced to a one line declaration by the prettyprinter. */

void ndecl15(char *dir)
{
  /* This declaration statement is spread over 5 lines. */
  int i,
    *k,
    l[10],
    (*pf)(),
    foo();

  i = 1;
}
