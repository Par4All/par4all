// pointer arithmetic - to be refined later with convex pointer values
int main()
{
  int a[10];
  int *p;
  int i,j;
  i = 3;
  j = 2;
  p = &a[0];
  p = p+i;
  p = p+j;
  p = p-i;
  p = p-j;
  return(0);
}
