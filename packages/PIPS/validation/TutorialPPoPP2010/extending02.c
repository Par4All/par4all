float extending02(int n, float a[n])
{
  int i;
  float s=0.;
  for(i=0;i<n;i++) {
    s += a[i]*a[i];
  }
  return s;
}
