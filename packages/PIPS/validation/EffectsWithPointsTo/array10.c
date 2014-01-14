/* FI looking for recursive calls: check subscript expressions
 *
 * For debugging, this test case has been broken into array10a, 10b
 * and 10c.
 */

double a[100];

int foo(int *p) {
  int b[100];
  p = &b[0];
  a[(*p)+1]= 2.;
  return 0;
}

int bar(int *p) {
 int b[100];
  p = &b[0];
  a[*p++]= 2.;
  return 0;
}

int toto(int *p) {
  int *q;  
  int b[100];
  p = &b[0];

  a[*(q=p++)]= 2.;
  return 0;
}
 

