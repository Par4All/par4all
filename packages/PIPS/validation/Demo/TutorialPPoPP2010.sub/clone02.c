int clone02(int n, int s)
{
  int r = n;
  if(s<0)
    r = n-1;
  else if(s>0)
    r = n+1;
  return r;
}

int main()
{
  int i = 1;
  i = clone02(i,-1);
  i = clone02(i,1);
  i = clone02(i,0);
}
