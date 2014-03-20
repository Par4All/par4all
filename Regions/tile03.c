// Test if the region of a 1-D tile is precise

int min(int a, int b)
{
  if(a<b) return a;
  else return b;
}

void tile03(int n, int ts, int a[n])
{
  int ti, i;
  for(ti=0; ti<n; ti += ts)
    for(i = ti; i < min(ti+ts, n); i++)
      a[i]=0;
}
