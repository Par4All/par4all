/* Expected result:
   A[i] is scalarized, the scalar is copied in from A[i],
   but not copied back into A[i].

   NOTE: no copy-out because the OUT region is empty
   in this context: no call site to scalarization09.

   See scalarization17 for fully working example.

 */

int SIZE = 10;
    
void scalarization09(double A[SIZE], double B[SIZE][SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++)
      A[i] = B[j][i] + A[i];
}


