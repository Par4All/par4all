/* Excerpt from hyantes */

#include <stdlib.h>

typedef double data_t;

typedef data_t town[3];
typedef struct {
    size_t nb;
    town *data;
} towns;

towns read_towns(const char fname[])
{
    towns the_towns = { 1 , malloc(sizeof(town)) };
    int curr = 0;
    double * px = &(the_towns.data[curr][0]);

    return the_towns;
}
