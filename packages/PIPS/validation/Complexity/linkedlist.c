#include <string.h>
#include <stdlib.h>

  struct Liste {
  char Chaine[16];

  struct Liste *pSuivant;

  };

int main ()
{


struct Liste *Nouveau;

struct Liste *Tete;

Tete = NULL;

// problem of sizeof instrinsic with structure 
// Nouveau = (struct Liste *)malloc(sizeof(struct Liste));

Nouveau->pSuivant = Tete;

Tete = Nouveau;

}
