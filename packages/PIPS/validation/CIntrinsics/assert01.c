/* assert example: In this example, assert is used to abort the program execution if datafile compares equal to 0, which happens when the previous call to fopen was not successful. */

#include <stdio.h>
#include <assert.h>

int main ()
{
  FILE * datafile;
  datafile=fopen ("file.dat","r");
  assert (datafile);

  fclose (datafile);

  return 0;
}
