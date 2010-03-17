/* To check the handling of non-boolean condition in C */

#include <stdio.h>

void if03()
{
  int n = 0;

  n = 0;

  if(n++)
    printf("n == true\n"); /* no */
  else
    printf("n == true\n"); /* yes */

  if(n++)
    printf("n == true\n"); /* yes */
  else
    printf("n == true\n"); /* no */
}
