/* To investigate the relationship between NULL and ANYWHERE */

#include <stdio.h>

void if12(char *filename)
{
  FILE *file;
  
  file = fopen(filename,"rb");
  if (file==NULL)
    printf("ouverture du fichier impossible\n");

  return;
}
