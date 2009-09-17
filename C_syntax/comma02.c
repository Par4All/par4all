/* To make sure that the comma expressions are not surrounded by
   useless parentheses */

#include <stdio.h>

int main()
{
  int i, j;

  i = (j = 1, 2);
  i = 3, j = 4;
}

