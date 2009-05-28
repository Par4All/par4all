/* Expected result:
   A[i] can be scalarized, but the scalar must be copied back into A[i]
   NOTES:
   - Bug: no copy-out
   - Additionally, one too many scalar is declared (__ld__1)
 */

#include <stdio.h>
#define n 10
    
void scalarization08(int x[n], int y[n][n])
{
  int i,j;
  for(i=0 ; i < n ; i++)
    for(j=0 ; j < n ;j++)
      x[i] = y[j][i] ;
}

int main(int argc, char **argv)
{
  int i;
  int x[n], y[n][n];

  for (i=0 ; i<n ; i++) {
    scanf("%d %d", &x[i], &y[i][i]);
  }
  scalarization08(x, y);

  printf("%d %d", x[n], y[n]);
}
