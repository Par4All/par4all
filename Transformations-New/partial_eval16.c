/* Check multiple assignments, function call sites and array
   subscript partial evaluation */

void foo(int i)
{
}

int main()
{
  int i = 1;
  int j = 2;
  int k = 3;
  int l;
  int m;
  int n;
  int a[10];

  i = j = k = 4;
  foo(i=j=k=5);
  foo(i-i);
  foo(a[i-i]);
  k = i*j+i*j;
  k = i*j+i*j+1;
  k = 1+i*j+i*j;
  k = m*n+m*n;
  k = m*n+m*n+1;
  k = 1+m*n+m*n;
  k += i-i;
  return i;
}
