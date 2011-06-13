/* pointer arithmetic with pointers to variable length array type, example from the C norme p84. */
#include<stdio.h>

int main()
{
  int n = 4, m = 3;
  int a[4][3];
  int (*p)[3];
  p = a;
  p += 1; // p == &a[1]
  (*p)[2] = 99; // a[1][2] = 99
  n = p-a; // n == 1
  printf("n = %d", n);
  return 0;

}
