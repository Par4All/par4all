
int main()
{
  int i, a[10], *p;
  
  p = &a[0];
  for(i=0; i<10; i++) {
    *p = i;
    p++;
  }
  
  return 0;
}
