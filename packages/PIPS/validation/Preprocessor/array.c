/* Bug in main function declaration */

void main ()
{
  int a[10];
  int nga;
  float b[10][20];
  nga = 2;
  a[nga] = 1;
  b[1][2] = 2;
}
