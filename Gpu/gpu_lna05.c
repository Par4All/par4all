// test case for GPU_LOOP_NEST_ANNOTATE
// when GPU_LOOP_NEST_ANNOTATE_PARALLEL is TRUE


int main(){
  float a[501];
  float b[501][501];
  float c[501][501][501];
  int i,j,k;

  // first loop nest : 
  // do par <- annotate
  //   do seq
   for(i = 0; i <= 123; i += 1)
      for(j = 0; j <= 498; j += 1)
	{
	  if (i==0)
	    a[i] = (float) i;
	  else
	    a[i] = a[i] + 1.0;
	}
	  
  // second loop nest : 
  // do par <- annotate whole loop nest
  //   do par
   for(i = 0; i <= 123; i += 1)
      for(j = 0; j <= 498; j += 1)
	b[i][j] = (float) i*j;

  // third loop nest : 
  // do par <- annotate whole loop nest
  //   do par
  //     do par
   for(i = 0; i <= 123; i += 1)
      for(j = 0; j <= 234; j += 1)
	for(k = 0; k <= 498; k += 1)
	  c[i][j][k] = (float) i*j;

  // fourth loop nest : 
  // do par <- annotate whole loop nest
  //   do par
   for(i = 10; i <= 508; i += 1)
      for(j = 20; j <= 518; j += 1)
	b[i-10][j-20] = (float) i*j;
   

  // fifth loop nest : 
  // do seq
  //   do par <- do not annotate
   for(i = 0; i <= 123; i += 1)
      for(j = 0; j <= 498; j += 1)
	b[i+1][j] = b[i][j] * 2.0;


  return 0;
}
