int dep(int a[100], int b[100], int c[100], int d[100])
{
  int i, j, k;
  //BEGIN_FPGA_dep_to_export
  for(i = 0; i < 100; i++)
    {
      a[i] = d[i];
      c[i] = a[i-1] + 1;
    }
  //END_FPGA_dep_to_export
  return 0;
}

int main(int argc, char* args)
{
  int a[100], b[100], c[100], d[100];

  dep(a, b, c, d);



  return 0;
}
