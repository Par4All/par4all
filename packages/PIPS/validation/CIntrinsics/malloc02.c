/* malloc example: string generator*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main ()
{
  int i,n;
  // FI: this declaration is incompatible with an assignment of buffer
  // via malloc()...
  //char buffer[50];
  char * buffer;

  printf ("How long do you want the string? ");
  scanf ("%d", &i);

  buffer = (char*) malloc (i+1);
  if (buffer==NULL) exit (1);

  for (n=0; n<i; n++)
    buffer[n]=rand()%26+'a';
  buffer[i]='\0';

  printf ("Random string: %s\n",buffer);
  memmove (buffer,"bonjour",7);
  free (buffer);


  return 0;
}
