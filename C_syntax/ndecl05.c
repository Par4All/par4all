/* Purpose: debug the new version of the parser which keep a statement
   for each declaration. Problem with declarations of function formal
   arguments: the second function declared loses its formal
   arguments... Was a prettyprint bug. */

void ndecl05(char *dir)
{
  void foo(int x), bar(int y);

}
