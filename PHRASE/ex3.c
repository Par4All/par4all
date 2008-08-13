int ex3(int argc, char* args)
{
  int x[100], u[100], z[100], y[100];
  int k, r, t;

  /* BEGIN_FPGA_TEST */
  for(k = 3, r = 2; k < 90 && r > 3; k++, r--)
    {
      x[k] = x[k-2] + x[k+3];
    }
  /* END_FPGA_TEST */

  return 0;
}
