// to test points_to_compare_cells()
int main()
{
  int *a[2], *q;
  int b[2];
  int c = 0;
  
  if(1)
    a[0] = &b[0];
  else
    a[0] = &b[1];
  
  a[1] = &c;
  q = a[0];
  return(0);
}
