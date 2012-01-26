/* For more information about this application excerpt, look at:

   http://hyantes.gforge.inria.fr

   Hyantes is a library to compute neighbourhood population potential with
   scale control. It is developed by the Mescal team from the Laboratoire
   Informatique de Grenoble, as a part of Hypercarte project. The
   Hypercarte project aims to develop new methods for the cartographic
   representation of human distributions (population density, population
   increase, etc.) with various smoothing functions and opportunities for
   time-scale animations of maps. Hyantes provides one of the smoothing
   methods related to multiscalar neighbourhood density estimation. It is
   a C library that takes sets of geographic data as inputs and computes a
   smoothed representation of this data taking account of neighbourhood's
   influence.
*/
// This case has been added to validate the generation of all effects
// from an address-of actual argument when it is a field of a structure
// (as the &the_towns.data[curr].latitude argument of the fscanf call)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define rangex 290
#define rangey 299
#define nb 2878

#ifdef USE_FLOAT
typedef float data_t;
#define INPUT_FORMAT "%f%*[ \t]%f%*[ \t]%f"
#define OUTPUT_FORMAT "%f %f %f\n"
#else
typedef double data_t;
#define INPUT_FORMAT "%lf%*[ \t]%lf%*[ \t]%lf"
#define OUTPUT_FORMAT "%lf %lf %lf\n"
#endif

typedef struct {
  data_t latitude;
  data_t longitude;
  data_t stock;
} town;

typedef struct {
    size_t n;
    town *data;
} towns;

towns read_towns(const char fname[])
{
    FILE * fd = fopen(fname,"r");
    size_t curr=0;
    char c;
    towns the_towns = { 1 , (town *) malloc(sizeof(town)) };
    fprintf(stderr,"begin parsing ...\n");

    while(!feof(fd))
    {
        if(the_towns.n==curr)
        {
            the_towns.n*=2;
            the_towns.data=(town *) realloc(the_towns.data,
					    the_towns.n*sizeof(town));
        }
        if(fscanf(fd,INPUT_FORMAT,&the_towns.data[curr].latitude,&the_towns.data[curr].longitude,&the_towns.data[curr].stock) !=3 )
        {
            while(!feof(fd))
            {
                c=(char)fgetc(fd);
                if(c=='\n' || c=='\r') break;
            }
        }
        else
        {
            the_towns.data[curr].latitude *=M_PI/180;
            the_towns.data[curr].longitude *=M_PI/180;
            ++curr;
        }
    }
    fclose(fd);
    the_towns.data=(town *) realloc(the_towns.data,curr*sizeof(town));
    the_towns.n=curr;
    fprintf(stderr,"parsed %zu towns\n",curr);
    /*
    for(curr=0;curr<the_towns.nb;curr++)
        fprintf(stderr,OUTPUT_FORMAT,the_towns.data[curr][0],the_towns.data[curr][1],the_towns.data[curr][2]);
    */
    return the_towns;
}

