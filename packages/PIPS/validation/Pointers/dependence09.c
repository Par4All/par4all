/*****************************************************************************
 * IX - STRUCTURE
 ****************************************************************************/
struct my_struct {
  int a[10];
  int b[10];
};
struct my_struct dependence09() {
  int *b, *a;
  int i;
  struct my_struct s;

  a = s.a; // a points_to s[1]
  b = s.b; // b points_to s[2]


  for ( i = 0; i < 10; i++ ) {
    a[i] = 0;
    b[i] = 1; // No dep
  }

  return s;
}
