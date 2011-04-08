//to test fopen, puts, gets,fprintf, fclose
#include <stdio.h>

int main ()
{
  FILE * pFile;
   int n;
  char name [100];

  pFile = fopen ("myfile.txt","w");
  for (n=0 ; n<3 ; n++)
    {
      puts ("please, enter a name: ");
      gets (name);
      fprintf (pFile,"Name %d %s\n",n,name);
    }
  fclose (pFile);

}
