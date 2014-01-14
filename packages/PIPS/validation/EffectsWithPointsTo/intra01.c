/* compute proper and cumulated effects for intra01
 * compute proper and cumulated effects with points-to for intra02
 * compute proper and cumulated pointer effects for intra03
 *
 * Used too. The code was replicated for each test case, including intra04.c
 *
 * Formal parameters pp and qq renamed fpp and fqq to simplify debug
 */

#include<stdio.h>

void bar(int **fpp, int **fqq) {
  *fpp = *fqq;
  printf("pointers copied");
  return;
}

void foo(){
  int i = 0 , j = 1,  *p = &i, *q = &j, **pp = &p, **qq = &q ;
  
  bar(pp, qq);
  return;
}
