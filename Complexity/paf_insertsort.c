/* Copied from Program Termination and Computational Complexity,
   C. Alias & al., March 2010, TR INRIA 7235

   Complexity: N^2/2+3N/2+1

   Problem:

   Bug: PIPS complexity does not work for complex for loops. Work for
   Molka Becher!
 */

void paf_insertsort(int len, int a[len])
{
  int i;

  for(i=1; i <len; i++) {
    int value = a[i], j;
    for(j=i+1; j>0 && a[j]>value; j--)
      a[j+1]=a[j];
    a[j+1] = value;
  }
  return;
}
