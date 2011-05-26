// basic test statements with subscripts
int main()
{
  int *a[2];
  int b = 0;
  int c = 1;
  if (1)
    a[0] = &b;
  else
    a[0] = &c;

  if(1)
    a[0] = &b;
  else
    a[0] = &b;
  return(0);
}
