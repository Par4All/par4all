/* FI looking for recursive calls */

#include <stdio.h>

double a[100];

int foo(int *p) {
  a[(*p)+1]= 2.;
  return 0;
}

int bar(int *p) {
  a[*p++]= 2.;
  return 0;
}
 

int toto(int *p) {
  int *q;  
  
  a[*(q=p++)]= 2.;
  return 0;
}
