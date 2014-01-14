/* Excerpt of hyantes 
 *
 * Typing bug for a points-to arc
 */

#include <stdio.h>

typedef double town[3];

typedef struct {
    size_t nb;
    town *data;
} towns;

void display(towns t, double xmin, double ymin,
	     double xmax, double ymax, double step)
{
    size_t rangex=( (xmax - xmin )/step ),
        rangey=( (ymax - ymin )/step );
    size_t i,j;
    for(i=0;i<rangex;i++)
    {
        for(j=0;j<rangey;j++)
            printf("%lf %lf %lf\n",
		   t.data[i*rangey+j][0],
		   t.data[i*rangey+j][1],
		   t.data[i*rangey+j][2]);
        printf("\n");
    }
    return;
}
