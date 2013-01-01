/* pi is written and ends up pointing to nowhere, but it does not
 * matter for the caller because pi is a copy of aipp.
 *
 * Note the importance of if(1) since the condition is *not*
 * interpreted. all updates are only potential (may information) for
 * the analysis.
 *
 * Thus the execution error cannot be safely exploited: maybe the else
 * branch is always executed.
 */

int formal_parameter01(int **pi)
{
  /* FI: I need the summary for the sequence, hence if(1)... which
     does not provide the summary for the sequence but a postcondition
     for a test... which is not at all satisfying because it does not
     fit the precondition of the test */
  if(1) {
    int ** q;
    int *i;
    int j;

    i = 0;
    q = pi;
    q++; // Incompatible with call site since pi points toward a scalar
    pi = &i;
    *pi = &j;
    *q = &j; // Incompatible with call site, see previous comment
  }
  return 0;
}

int main()
{
  int i, *ip, **aipp;
  ip = &i;
  aipp = &ip;
  i = formal_parameter01(aipp);
  return 0;
}
