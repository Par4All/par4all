int main()
{ 
  float n=100.0;
  float m = 1.0;
  int dec = (int) n;
  int A[dec];
  int k,j,z;

  for (k=dec;k>=1;k--)
    for (j=(int)m;j<=k-1;j++)
      if (A[j]>A[j+1])
	{
	  z=(int)A[j];
	  A[j]=A[j+1];
	  A[j+1]=z;
	}

  return A;
}
