// test case for GPU_LOOP_NEST_ANNOTATE
// when GPU_LOOP_NEST_ANNOTATE_PARALLEL is TRUE


int main(){
  float a[501];
  float b[501][501];
  float c[501][501][501];
  int i,j,k;

   for(i = 0; i <= 123; i += 2)
      for(j = 0; j <= 498; j += 1)
	{
	  if (i==0)
	    a[i] = (float) i;
	  else
	    a[i] = a[i] + 1.0;
	}


  return 0;
}
