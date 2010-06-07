/* Copied from Program Termination and Computational Complexity,
   C. Alias & al., March 2010, TR INRIA 7235. I guess copied from a
   paper by Sipma

   Complexity: N^2+2N+3

   Problem: while loop are used. I do not know if PIPS can recover
   them automatically. Apparently the external loop is recovered but
   not converted in to a do loop (Serge Guelton?).

   Bug:
 */

void paf_sipmabubble(int n, int A[n])
{
  int i = n;
  while(i>=0) {
    int j = 0;
    while(j<=i-1) {
      if(A[j]>A[j+1]) {
	int tmp = A[j];
	A[j] = A[j+1];
	A[j+1] = tmp;
      }
      j++;
    }
    i--;
  }
  return;
}
