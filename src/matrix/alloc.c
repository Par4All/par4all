 /* package matrix */

#include <stdio.h>
#include <sys/stdtypes.h> /*for debug with dbmalloc */
#include <malloc.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrix.h"


Pmatrix matrix_new(n,m)
int n,m;
{ 
    Pmatrix a = (Pmatrix) malloc(sizeof(Smatrix));
    a->denominator = VALUE_ONE;
    a->number_of_lines = n;
    a->number_of_columns = m;
    a->coefficients = (Value *) malloc(sizeof(Value)*((n*m)+1));
    return (a);
}
