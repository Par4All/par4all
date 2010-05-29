// The declaration of array c is useless, but not the expression used
// to size it

int c_clean_declarations06(int x)
{
  int a[x];
  int b = sizeof(a);
  int c[b++];
  return b;
}
