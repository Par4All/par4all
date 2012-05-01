/* Loop based on pointer arithmetic */

#include <stdio.h>

#define N 10

int t[N] = {1,2,0,11,0,12,13,14,0,4};

int main()
{
  int *pdeb,*pfin,*p;

  pdeb = &t[0];      /*   repère le premier élément de t   */
  pfin = &t[N-1];    /*   repère le dernier élément de t   */

  for (p = pdeb; p <= pfin; p++)
    if (*p == 0) printf("%d ",(p - pdeb));
  printf("\n");
  return 0;
}

