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

#include <p4a_accel.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define rangex 290
#define rangey 299
#define nb 2878
#define M_PI 3.14

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


// Prototypes for the two kernels of jacobi
P4A_wrapper_proto(kernel_wrapper,data_t xmin, data_t ymin, data_t step,data_t range, P4A_accel_global_address town pt[rangex][rangey], P4A_accel_global_address town t[nb]);

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
    return the_towns;
}


void launch_kernel(data_t xmin, data_t ymin, data_t step,data_t range, town pt[rangex][rangey], town t[nb]) 
{
  //P4A_call_accel_kernel_2d(kernel_wrapper, rangex, rangey, xmin,ymin,step,range,pt,t);
  P4A_call_accel_kernel_2d(kernel_wrapper, rangex, rangey, xmin,ymin,step,range,pt,t);
}


void run(data_t xmin, data_t ymin, data_t step,data_t range, town pt[rangex][rangey], town t[nb])
{
    fprintf(stderr,"begin computation ...\n");

    launch_kernel(xmin,ymin,step,range,pt,t);

    fprintf(stderr,"end computation ...\n");
}

void display(town pt[rangex][rangey])
{
    size_t i,j;
    for(i=0;i<rangex;i++)
    {
        for(j=0;j<rangey;j++)
            printf(OUTPUT_FORMAT,pt[i][j].latitude, pt[i][j].longitude, pt[i][j].stock);
        printf("\n");
    }
}

int main(int argc, char * argv[])
{
  P4A_init_accel;
  if(argc != 6) return 1;
  {
    town pt[rangex][rangey];
    towns t = read_towns(argv[1]);
    data_t xmin = atof(argv[2])*M_PI/180.,
      ymin =atof(argv[3])*M_PI/180.,
      step=atof(argv[4])*M_PI/180.,
      range=atof(argv[5]);
    

    P4A_accel_timer_start;
    town (*p4a_pt)[rangex][rangey];
    P4A_accel_malloc((void **) &p4a_pt, sizeof(pt));
    //P4A_copy_to_accel(sizeof(pt), pt, p4a_pt);
    

    town (*p4a_t)[nb];
    P4A_accel_malloc((void **) &p4a_t, nb*sizeof(town));
    P4A_copy_to_accel(nb*sizeof(town), t.data, p4a_t);
    
    double copy_time = P4A_accel_timer_stop_and_float_measure();

    P4A_accel_timer_start;
    run(xmin,ymin,step,range, *p4a_pt, *p4a_t);
    
    double execution_time = P4A_accel_timer_stop_and_float_measure();
    fprintf(stderr, "Time of execution : %f s\n", execution_time);
    fprintf(stderr, "GFLOPS : %f\n",
	    4e-9/execution_time*(rangex)*(rangey));
    
    P4A_accel_timer_start;
    P4A_copy_from_accel((size_t)sizeof(pt), (void *)pt, (void *)p4a_pt);
    copy_time += P4A_accel_timer_stop_and_float_measure();
    fprintf(stderr, "Time of copy : %f s\n", copy_time);

    P4A_accel_free(p4a_pt);
    P4A_accel_free(p4a_t);

    display(pt);
  }
  P4A_release_accel;
  return 0;
}
