#include <stdio.h>

int guard(int i, int n, int a[n], int b[n])
{
   register int a_0 = a[i];

   if (i<n)
      b[i] = a_0/2+a_0/3;
}

int main(int argc, char **argv)
{
   int n = 100;
   int a[n], b[n];
   {
      int i;

      for(i = 0; i <= n-1; i += 1)
         a[i] = i;
   }
   {
      int i;

      for(i = 0; i <= n-1; i += 1)
         guard(i, n, a, b);
   }
   {
      int i;

      for(i = 0; i <= n-1; i += 1)
         printf("%d\n", b[i]);
   }
}
