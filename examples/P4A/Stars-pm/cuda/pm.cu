#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stars-pm-cuda.h"

void printUsage(int argc, char **argv) {
  printf("Usage : \n\n%s file\n", argv[0]);
}

int main(int argc, char **argv) {

  /******************************************************
   *                    ALLOCATIONS
   ******************************************************/
  float mp[NP][NP][NP]; // Mass for each particle
  coord pos[NP][NP][NP]; // Position (x,y,z) for each particle
  coord vel[NP][NP][NP]; // Velocity (x,y,z) for each particle

  int data[NP][NP][NP]; //
  int histo[NP][NP][NP]; //


  float time;
  int npdt = 0;
  float dt = DT / 2; // Will be 1/2 only for the first iteration

	char * icfile=NULL;
	
  /******************************************************
   *               GET INITIAL CONDITIONS
   ******************************************************/
  icfile = argv[1];
  if(NULL == icfile) {
    printUsage(argc, argv);
    exit(-1);
  }

  // Read input file
  readDataFile(mp, pos, vel, icfile);


  /******************************************************
   *               INITIALIZE GPU MEMORY
   ******************************************************/

  float *posd; // Position
  toolTestExec(cudaMalloc((void **)&posd,sizeof(float)*3*NPART));
  cudaMemcpy(posd,pos,sizeof(float)*NPART*3,cudaMemcpyHostToDevice);

  float *veld; // Velocity
  toolTestExec(cudaMalloc((void **)&veld,sizeof(float)*3*NPART));
  cudaMemcpy(veld,vel,sizeof(float)*NPART*3,cudaMemcpyHostToDevice);


  float *densd; // Densité pour chaque cellule
  toolTestExec(cudaMalloc((void**)&densd,sizeof(float)*NPART));
  float *forced; // Densité Temporaire interpolée successivement sur x,y, et z
  toolTestExec(cudaMalloc((void**)&forced,sizeof(float)*NPART));

  int *datad; //
  toolTestExec(cudaMalloc((void**)&datad,sizeof(int)*NPART));
  int *histod; //
  toolTestExec(cudaMalloc((void**)&histod,sizeof(int)*NPART));

  potential_init_plan(NULL); // Init fft stuff

#ifdef _GLGRAPHICS_
      graphic_gldraw(argc, argv, pos); // Initialize Opengl
#endif


  /******************************************************
   *                  MAIN LOOP !!
   ******************************************************/
  for (time = 0; time <= TMAX; time += DT) {
    if(0 == npdt % MODDISP) {
      puts("**********************************");
      printf("Time= %5.2e Npdt= %d\n", time, npdt);
    }

    // Stage 1 : discretization of particles position
    discretization((coord (*)[NP][NP])posd, (int (*)[NP][NP])datad);

    // Stage 2 : computing density, sequential histogramming pass
    cudaMemcpy(data, datad, sizeof(int) * NPART, cudaMemcpyDeviceToHost);
    histogram(data,histo);
    cudaMemcpy(histod, histo, sizeof(int) * NPART, cudaMemcpyHostToDevice);

#ifdef _GRAPHICS_
    if(0 == npdt % MODDISP) {
      // GTK graphic output
      graphic_draw(argc, argv, histo);
    }
#endif

    // Stage 3 : computing potential from density
    potential((int (*)[NP][NP])histod, (float (*)[NP][NP])densd, 0, mp );

    // Stage 4 : computing force from potential in each direction, and apply it on velocity
    forcex((float (*)[NP][NP])densd, (float (*)[NP][NP])forced);
    updatevel((coord (*)[NP][NP])veld, (float (*)[NP][NP])forced, (int (*)[NP][NP])datad, 0, dt);

    forcey((float (*)[NP][NP])densd, (float (*)[NP][NP])forced);
    updatevel((coord (*)[NP][NP])veld, (float (*)[NP][NP])forced, (int (*)[NP][NP])datad, 1, dt);

    forcez((float (*)[NP][NP])densd, (float (*)[NP][NP])forced);
    updatevel((coord (*)[NP][NP])veld, (float (*)[NP][NP])forced, (int (*)[NP][NP])datad, 2, dt);

    // Final stage : updating position using new velocity
    updatepos((coord (*)[NP][NP])posd, (coord (*)[NP][NP])veld);

    // Timestep is 1/2 DT for first iteration
    dt = DT;

#ifdef _GLGRAPHICS_
    cudaMemcpy(posd, pos, sizeof(coord) * NPART, cudaMemcpyDeviceToHost);
    graphic_glupdate(pos);
#endif
#ifdef _DUMPPOS_
    cudaMemcpy(posd, pos, sizeof(coord) * NPART, cudaMemcpyDeviceToHost);
    dump_pos(pos,npdt);
#endif

    npdt++;
  }
	
  potential_free_plan();


	puts("-----------------------------");
	puts("Finished");
	puts("-----------------------------");
	
	return 0;
}
