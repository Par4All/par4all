/* To check pointer references
 *
 * Same as pointer_reference02(), but for a comparison
 *
 * gcc does not give type information when an assignment mistake is
 * made but it does give type information when a call site is wrong.
 */

#include <stdio.h>

int pointer_reference03(char **p)
{
  char * q = p[1];
  char * r;
  r = *(p+3);
  printf("\nq=%p\nr=%p\n", q, r);
  return q-r;
}

int main()
{
  // char tab[10][10];
  // type of "tab" according to gcc: ‘char (*)[10]’

  char * ptab[10];

  char tab1[10];
  ptab[0] = tab1;
  char tab2[10];
  ptab[1] = tab2;
  char tab3[10];
  ptab[2] = tab3;
  char tab4[10];
  ptab[3] = tab4;

  printf("tab1=%p\ntab2=%p\ntab3=%p\ntab4=%p\n", tab1, tab2, tab3, tab4);

  int i = pointer_reference03(ptab);

  printf("%d\n", i);
  return 0;
}
