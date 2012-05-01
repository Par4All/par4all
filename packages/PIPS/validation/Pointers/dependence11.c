/*****************************************************************************
 * XI - STRUCTURE AS PARAM
 ****************************************************************************/
struct my_struct {
  int a[10];
  int *b;
};
void dependence11( struct my_struct *s ) {
  int *b, *a;
  int i;
  a = s->a; // a points_to s[0][1]
  b = s->b; // b points_to s[0][2]?


  for ( i = 0; i < 10; i++ ) {
    a[i] = 0;
    b[i] = 1; // No dep
  }

}
