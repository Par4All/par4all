#include <stdio.h>

int main ()
{
  FILE* fp;
  size_t n;
  char buf[200];
  fp = fopen ("file.txt", "rb");
  n = fread (&(buf[0]),sizeof(double), 50, fp);
  if (n == -1)
    printf("fread failed");
  fclose (fp);
  return 0; 
}

