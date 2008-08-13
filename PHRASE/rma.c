int rma(int a[100][100], int b[100], int c[100], int d[100])
{
  int i, j, k;
  //BEGIN_FPGA_rma_to_export
  for(i = 0; i < 101; i++)
    {
      c[i*i] = c[i*i] + d[i];
      for(j = 0; j < 101; j++)
	{
	  c[i*i] = c[i*i] + a[i][j] * b[j];
	  d[i+1] = d[i] + b[j];
	}
    }
  //END_FPGA_rma_to_export
  return 0;
}

int main(int argc, char* args)
{
  int a[100][100], b[100], c[100], d[100];

  rma(a, b, c, d);



  return 0;
}
