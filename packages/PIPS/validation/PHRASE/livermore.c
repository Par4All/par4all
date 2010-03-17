int livermore(int argc, char* args)
{
  int b[100][100], w[100], a[5];
  int i, j, k;

  /* BEGIN_FPGA_TEST */
  for ( i=1 ; i<100 ; i++ )
    {
      for ( k=0 ; k<i ; k++ )
	{
	  w[i] += b[k][i] * w[(i-k)-1] + a[0];
	}
    }
  /* END_FPGA_TEST */

  return 0;
}
