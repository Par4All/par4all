/* Check multiple assignments, function call sites and array
   subscript partial evaluation. */

void foo(int i)
{
}

int main()
{
  int i;
  int k = 2;
  int j;
  int m;
  int a[10];

  i = 1, j = 2*(m-m), k = 3*(m-m);
  i = k += j-j;
  a[i-i] = 1, a[j-j] = 2, a[k-k] = 3;
  k -= j-j;
  k *= j-j;
  k /= j+1-j;
  k <<= j-j;
  k >>= j-j;
  k |= j*m-j*m;
  i++ + j-j;
  return i;
}
