void array_addition01(float**  a1,
		      float**  a2,
		      int m, int n)
{
  int i,j ;
  for( i=0;i<m;i++ ) {
    for( j=0;j<n;j++ ) {	
      a2[i][j] = a1[i][j] +2;
    }
  }
  return ;
}
