/*int muladd(int a[100][100], int b[100], int c[100], int d[100])
{
  int i, j, k;
  //BEGIN_FPGA_muladd_HRE
  for(i = 0; i < 100; i++)
    {
      c[i] = c[i] + c[i] + d[2*i+1];

      for(j = 0; j < 100; j++)
	{
	  d[2*i] = c[i] + a[i][j] * b[j];
	  c[i] = a[i][j] * b[j];
	  c[i] = a[i][j] * b[j] + d[2*i+1];
          b[j] = c[i] * d[2*i] + 3;
	}

      c[i] = c[i] + d[2*i];
      d[2*i-1] = d[2*i-1] + 2;

      for(k = 0; k < 100; k++)
	{
	  d[2*i-1] = c[i] + a[i][k] * b[k];
	  c[i] = a[i][k] * b[k];
	  c[i] = a[i][k] * b[k] + d[2*i+1] + d[2*i-1];
          b[k] = c[i] * d[2*i] + 3;
	}

      d[2*i-1] = c[i];


    }
  //END_FPGA_muladd_HRE
  return 0;
}
*/

void muladd(int a[100][100], int b[100],
            int c[100], int d[100])
{
  int i, j, k;

  //BEGIN_FPGA_muladd_HRE
  for(i = 0; i < 100; i++)
    {
      c[i] = c[i] + d[i];

      for(j = 0; j < 100; j++)
	{
	  c[i] = c[i] + a[i][j] * b[j];
	}
    }
  //END_FPGA_muladd_HRE

  return;
}

int main(int argc, char* args)
{
  int a[100][100], b[100], c[100], d[100];

  muladd(a, b, c, d);



  return 0;
}
