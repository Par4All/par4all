/* Test how comments are kept if there are expressions in parenthesis in
   the for loop clause.

   Weird isn't it?

   I think now (RK, 2011/02/05 :-) ) that in a source-to-source compiler
   comments should appear *explicitly* in the parser and not be dealt by
   some side effects in the lexer as now in PIPS
*/
int main()
{
  int i, j = 0;

  // This is an overcommented for loop :-)
  for(/* before init */i = (0)/* after init */; /* before cond */(i < (5))/* after cond */;/* before inc */ (i++)/* after inc */) /* between the clause and block */ {
    /* loop body */
    j++;
    /* end of loop body */
  }
  /* after the loop */

  return 0;
}
