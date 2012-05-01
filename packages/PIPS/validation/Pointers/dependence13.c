/*****************************************************************************
 * IX - STRUCTURE
 ****************************************************************************/
struct my_struct {
  int a[10];
  int b[10];
};
struct my_struct dependence13() {
  int *b, *a;
  int i;
  struct my_struct s;

  return s;
}
