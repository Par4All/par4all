/* The pointer assignment is not translated fully acurately in the
 * main function. When "assignment" is analyzed, the value assigned
 * can be either a defined value or a NULL pointer.
 *
 * When the call site is analyzed, it is clear that "(**t2).ip2" is
 * not NULL.
 *
 * However, the alias analysis is not performed and there is no way to
 * induce that, hence, (**t1).ip1 is also different from NULL.
 */

#include <stdio.h>
#include <stdlib.h>

struct foo {int * ip1; int * ip2;} ;

 void assignment(struct foo** t1, struct foo** t2) 
 {
   (**t1).ip1 =(**t2).ip2;
 } 

int main() {
  struct  foo s1;
  struct foo s2;
  struct  foo** ppt, **pps ;
  struct  foo* pt = &s1, *ps = &s2;
  int i11 = 1, i12 = 2, i21 = 3, i22 = 4;

  s1.ip1 = &i11;
  s1.ip2 = &i12;
  s2.ip1 = &i21;
  s2.ip2 = &i22;
  
  *pt = s2;
  ppt = &pt;
  pps = &ps;
  assignment(pps, ppt);
  
  return 0;
}
