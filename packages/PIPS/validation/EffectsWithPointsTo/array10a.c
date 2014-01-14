/* FI looking for recursive calls: first third of array10, foo
 *
 * A dereferencement is left in an effect because it is hidden in the
 * subscript expression...
 */

double a[100];

int foo(int *p) {
  int b[100];
  p = &b[0];
  a[(*p)+1]= 2.;
  return 0;
}
 

