/* Expected result:
   A[i] can be scalarized, but the scalar must be copied back into A[i]
   NOTES:
   - Bug: no copy-out
   - Additionally, one too many scalar is declared (__ld__1)
 */

int SIZE = 10;
    
double get(double f[SIZE],int i) {
  return f[i];
}

void scalarization08(double A[SIZE], double B[SIZE][SIZE])
{
  int i,j;
  for(i=0;i<SIZE;i++)
    for(j=0;j<SIZE;j++)
      A[i] = B[j][i] ;
}


