/* Partial evaluation of declarations */

int main()
{
  int n = 2;
  int j = n++;
  int k = n+1;
  int a[n];
  int b[10];
  return a[j-1];
}
