/*
 * file for hyantes-static-99.c
 */
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef double data_t;
typedef struct {
   data_t latitude;
   data_t longitude;
   data_t stock;
} town;
typedef struct {
   size_t n;
   town *data;
} towns;

towns read_towns(const  char fname[]);



void run(data_t xmin, data_t ymin, data_t xmax, data_t ymax, data_t step, data_t range, town pt[290][299], town t[2878]);


void display(town pt[290][299]);


int main(int argc, char *argv[]);
towns read_towns(const  char fname[])
{
   FILE *fd = fopen(fname, "r");
   size_t curr = 0;
   char c;
   towns the_towns = {1, (town *) malloc(sizeof(town))};
   fprintf(stderr, "begin parsing ...\n");

   while (!feof(fd)) {
      if (the_towns.n==curr) {
         the_towns.n *= 2;

         the_towns.data = (town *) realloc(the_towns.data, the_towns.n*sizeof(town));
      }
      if (fscanf(fd, "%lf%*[ \t]%lf%*[ \t]%lf", &(the_towns.data)[curr].latitude, &(the_towns.data)[curr].longitude, &(the_towns.data)[curr].stock)!=3) {
l99999:         ;
         if (!!feof(fd)) goto break_2;
         c = (char) fgetc(fd);
         if (c=='\n'||c=='\r') {
         }
         else {
            goto l99999;
         }
break_2:         ;
      }
      else {
         (the_towns.data)[curr].latitude *= 3.14159265358979323846/180;
         (the_towns.data)[curr].longitude *= 3.14159265358979323846/180;
         ++curr;
      }
   }
   fclose(fd);
   the_towns.data = (town *) realloc(the_towns.data, curr*sizeof(town));
   the_towns.n = curr;
   fprintf(stderr, "parsed %zu towns\n", curr);
   /*
       for(curr=0;curr<the_towns.nb;curr++)
           fprintf(stderr,OUTPUT_FORMAT,the_towns.data[curr][0],the_towns.data[curr][1],the_towns.data[curr][2]);
       */
   return the_towns;
}
void run(data_t xmin, data_t ymin, data_t xmax, data_t ymax, data_t step, data_t range, town pt[290][299], town t[2878])
{
   size_t i, j, k;

   fprintf(stderr, "begin computation ...\n");

#pragma omp parallel for private(k, j)
   for(i = 0; i <= 289; i += 1)
      for(j = 0; j <= 298; j += 1) {
         pt[i][j].latitude = (xmin+step*i)*180/3.14159265358979323846;
         pt[i][j].longitude = (ymin+step*j)*180/3.14159265358979323846;
         pt[i][j].stock = 0.;
         for(k = 0; k <= 2877; k += 1) {
            data_t tmp = 6368.*acos(cos(xmin+step*i)*cos(t[k].latitude)*cos(ymin+step*j-t[k].longitude)+sin(xmin+step*i)*sin(t[k].latitude));
            if (tmp<range)
               pt[i][j].stock += t[k].stock/(1+tmp);
         }
      }
   fprintf(stderr, "end computation ...\n");
}
void display(town pt[290][299])
{
   size_t i, j;
   for(i = 0; i <= 289; i += 1) {
      for(j = 0; j <= 298; j += 1)
         printf("%lf %lf %lf\n", pt[i][j].latitude, pt[i][j].longitude, pt[i][j].stock);
      printf("\n");
   }
}
int main(int argc, char *argv[])
{
   if (argc!=8)
      return 1;
   {
      town pt[290][299];
      towns t = read_towns(argv[1]);
      
      
      
      
      
      data_t xmin = atof(argv[2])*3.14159265358979323846/180., ymin = atof(argv[3])*3.14159265358979323846/180., xmax = atof(argv[4])*3.14159265358979323846/180., ymax = atof(argv[5])*3.14159265358979323846/180., step = atof(argv[6])*3.14159265358979323846/180., range = atof(argv[7]);
      run(xmin, ymin, xmax, ymax, step, range, pt, t.data);
      display(pt);
   }
   return 0;
}
