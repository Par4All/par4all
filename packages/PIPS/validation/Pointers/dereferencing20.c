/* Excerpt from hyantes, more extended than dereferencing18.c and
   dereferencing19.c */

#include <stdio.h>
#include <stdlib.h>

typedef double data_t;
#define INPUT_FORMAT "%lf%*[ \t]%lf%*[ \t]%lf"

typedef data_t town[3];
typedef struct {
    size_t nb;
    town *data;
} towns;

towns read_towns(FILE * fd)
{
    towns the_towns = { 1 , malloc(sizeof(town)) };
    int curr = 0;
    if(curr)
      the_towns.data = realloc(the_towns.data,the_towns.nb*sizeof(town));
    double * px = &(the_towns.data[curr][0]);

    // This fscanf shows bugs in points-to analysis
    if(fscanf(fd,INPUT_FORMAT,&the_towns.data[curr][0],&the_towns.data[curr][1],&the_towns.data[curr][2]) !=3 )
      ;

    return the_towns;
}
