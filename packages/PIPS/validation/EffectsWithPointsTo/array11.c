/* FI looking for recursive calls
 *
 * A simple variation on array10c.c
 */

double a[100];

int titi(int *p) {
  int *q;  
  int b[100];
  p = &b[0];

  a[*(q=p)]= 2.;
  return 0;
}
 

