/*****************************************************************************
 * X - MULTIPLE PATH
 ****************************************************************************/
void dependence10( int flag, int *a ) {
  int *p;
  int b[10];
  int i;

  // Two paths
  if( flag)
    p = a;
  else
    p = b;

  // p may points to a[0] or b[0];

  for ( i = 0; i < 10; i++ ) {
    p[i] = 0; // We may write a or b, but only a is visible from outside
  }
}
