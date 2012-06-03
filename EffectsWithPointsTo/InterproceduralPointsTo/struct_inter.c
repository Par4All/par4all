#include <stdio.h>
#include <stdlib.h>

struct foo {int * ip1; int * ip2;} ;

 void assignment(struct foo t1, struct foo t2) 
 { 
   t1 = t2; 
 } 

int main() {
  struct  foo s1;
  struct foo s2;
  struct  foo* ps = &s1;
  int i11 = 1, i12 = 2, i21 = 3, i22 = 4;

  s1.ip1 = &i11;
  s1.ip2 = &i12;
  s2.ip1 = &i21;
  s2.ip2 = &i22;

  *ps = s2;
  assignment(s1, s2);
  s1 = s2;
  //printf("%d\n", *(s1.ip1));

  return 0;
}
