/* Excerpt of hantes.c used to debug struct initialization.
 */

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
    return the_towns;
}
