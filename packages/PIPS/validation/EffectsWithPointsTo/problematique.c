/* Calcul des effets pour des tableaux dynamiques */
#include<stdlib.h>
int main()
{
     int *a, *b, *c;
     int i;
     i = 0;
/* Allocation dynamiques des tableaux */
     a = (int*)malloc(20*sizeof(int));
     b = (int*)malloc(20*sizeof(int));
     c = (int*)malloc(20*sizeof(int));
/* Creation d'alias entre a et b */
     a = b;

/* Suppression de l'ancien alias entre a et b, creation d'un nouveau
 * entre a et c */
     a = c;

/* Initialisation du tableau b */
     for(i=0; i<20; i++)
	  b[i] =i;

/* Initialisation du tableau c */
     for(i=0; i<20; i++)
	  c[i] = 1;
/* Initialisation du tableau a */
     for(i=0; i<20; i++)
	  a[i] = b[i];
  return 0;
}
