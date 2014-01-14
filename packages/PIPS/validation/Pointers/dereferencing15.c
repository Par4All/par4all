/* Excerpt of dereferencing13, to chase a bug with the reference to w
 *
 *
 *
 */

double dereferencing15()
{
  double ** w;
  int i, *pi=&i;

  // w is used unitialized and this is not detected by the points-to analysis
  **(w+(i=2)) = 4.;
  return 0.;
}

