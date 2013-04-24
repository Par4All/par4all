// we have l<=i and i<=k
// We want at the end, l<=k

void main()
{
  int i, j, k, l, m;
  int *p;
  
  m=10;
  
  if (l<=i && i<=k)
  {
    if (j==0)
      p = &i;
    else
      p = &j;
    
    //without points-to informaion, we lost everything
    //with points-to informaion, we have l<=k
    *p=10;
    
    return;
  }
}
