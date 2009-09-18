/* To check the handling of non-boolean condition in C */

#include <stdio.h>

void if02()
{
  int j = -1;
  int n = 2;
  int i = 0;

  if(n)
    printf("n == true\n"); /* yes */
  else
    printf("n == true\n"); /* no */

  if(i)
    printf("i == true\n"); /* no */
  else
    printf("n == true\n"); /* yes */

  if(j)
    printf("j == true\n"); /* yes */
  else
    printf("n == true\n"); /* no */
}
