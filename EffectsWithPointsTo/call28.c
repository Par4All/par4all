/* To debug issues linked to on-demand generation of points-to
 *
 * The derefencement of q in call28_caller() is hidden in
 * call28(). The points-to information computed for call28_caller() is
 * currently insufficient for the proper effect computation.
 *
 * Use different names for formal and actual arguments: it makes
 * debugging much easier.
 */

void call28(int * qq)
{
  *qq = 1;
  return;
}

int call28_caller(int * q)
{
  call28(q);
  return 0;
}
