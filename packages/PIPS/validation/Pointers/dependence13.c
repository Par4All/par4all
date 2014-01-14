/*****************************************************************************
 * IX - STRUCTURE
 ****************************************************************************/

/* struct are passed and returned by copy, and copy of undefined
   values is allowed by the C standard. This piece of code is
   allright. */

struct my_struct {
  int a[10];
  int b[10];
  int *p;
};

struct my_struct dependence13() {
  int *b, *a;
  int i;
  struct my_struct s;

  return s;
}
