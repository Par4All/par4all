#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "matrice.h"
#include "genC.h"
#include "misc.h"

void hyperplane_direction(h,n)
int *h;
int n;
{
    int i;
    int n_read;

    /*lecteure du h */
    assert(n>=1);
    debug(8,"hyperplane_direction","\nLecteur du h");
    printf("Vecteur de direction :\n");

    for( i = 0; i<n; i++)
	n_read = fscanf(stdin," %d",h+i);
}
