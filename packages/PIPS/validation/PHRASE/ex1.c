int ex1(int argc, char* args)
{
  int x[100], u[100], z[100], y[100];
  int k, r, t;

  /* BEGIN_FPGA_TEST */
  for(k = 0; k < 100; k++)
    {
      x[k] = u[k] + r*( z[k] + r*y[k] ) +
	t*( u[k+3] + r*( u[k+2] + r*u[k-1] ) +
	    t*( u[k+6] + r*( u[k+5]*u[k+5] + r*u[k+4] ) ) );
    }
  /* END_FPGA_TEST */

  return 0;
}
