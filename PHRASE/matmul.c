int matmul(int a[100][100], int b[100][100], int c[100][100])
{
  int i, j, k;

  //BEGIN_FPGA_matmul_to_export
  for(i = 0; i < 101; i++)
    {
      for(j = 0; j < 101; j++)
	{
	  for(k = 0; k < 101; k++)
	    {
	      a[i][j] = a[i][j] + b[i][k] * c[k][j];
	    }
	}
    }
  //END_FPGA_matmul_to_export

  return 0;
}

/*int matmul(int a[100][100], int b[100][100], int c[100][100])
{
  int i, j, k;

  //BEGIN_FPGA_matmul_to_export
  for(i = 0; i < 100; i++)
    {
      for(k = 0; k < 100; k++)
	{
	  for(j = 0; j < 100; j++)
	    {
	      a[i][j] = a[i][j] + b[i][k] * c[k][j];
	    }
	}
    }
  //END_FPGA_matmul_to_export

  return 0;
}
*/
int main(int argc, char* args)
{
  int a[100][100], b[100][100], c[100][100];

  matmul(a, b, c);



  return 0;
}
