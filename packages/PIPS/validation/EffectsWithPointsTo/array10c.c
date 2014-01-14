/* FI looking for recursive calls: third third of array10, toto */

double a[100];

int toto(int *p) {
  int *q;  
  int b[100];
  p = &b[0];

  a[*(q=p++)]= 2.;
  return 0;
}
 

