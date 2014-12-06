void unroll21()
{
  int i, a[2];
bar:for(i=0;i<2;i++)
    {
      struct s {int f;};
      struct s j;
      j.f = i+2;
      a[i]=j.f;
    }
}
