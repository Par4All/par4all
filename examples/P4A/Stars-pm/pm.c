#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <fftw3.h>

#include "varglob.h"
#include "io.h"

#ifdef _GRAPHICS_
#include "graphics.h"
#endif
#ifdef _GLGRAPHICS_
#include "glgraphics.h"
#endif

#define max(x,y) ((x>y)?x:y)

void forcex(float pot[NCELL][NCELL][NCELL], float fx[NCELL][NCELL][NCELL]) {
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        fx[i][j][k] = (pot[(i + 1) & (NCELL - 1)][j][k] - pot[(i - 1) & (NCELL
            - 1)][j][k]) / (2. * DX);
      }
    }
  }
}

//========================================================================
void real2Complex(float cdens[NCELL][NCELL][NCELL][2],
                  float dens[NCELL][NCELL][NCELL]) {
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        cdens[i][j][k][0] = dens[i][j][k];
        cdens[i][j][k][1] = 0;
      }
    }
  }
}

void complex2Real(float cdens[NCELL][NCELL][NCELL][2],
                  float dens[NCELL][NCELL][NCELL]) {

  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        dens[i][j][k] = cdens[i][j][k][0];
      }
    }
  }
}

//========================================================================
void fft_laplacian7(float field[NCELL][NCELL][NCELL][2]) {
  int i, j, k;
  float i2, j2, k2;

  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        i2 = (i > NCELL / 2. ? sinf(M_PI / NCELL * (i - NCELL)) * sinf(M_PI
            / NCELL * (i - NCELL)) : sinf(M_PI / NCELL * i) * sinf(M_PI / NCELL
            * i));
        j2 = (j > NCELL / 2. ? sinf(M_PI / NCELL * (j - NCELL)) * sinf(M_PI
            / NCELL * (j - NCELL)) : sinf(M_PI / NCELL * j) * sinf(M_PI / NCELL
            * j));
        k2 = (k > NCELL / 2. ? sinf(M_PI / NCELL * (k - NCELL)) * sinf(M_PI
            / NCELL * (k - NCELL)) : sinf(M_PI / NCELL * k) * sinf(M_PI / NCELL
            * k)) + i2 + j2;
        k2 += (k2 == 0);

        field[i][j][k][0] = field[i][j][k][0] * G * M_PI * DX * DX / k2 / NCELL
            / NCELL / NCELL;
        field[i][j][k][1] = field[i][j][k][1] * G * M_PI * DX * DX / k2 / NCELL
            / NCELL / NCELL; // FFT NORMALISATION

      }
    }
  }

  field[0][0][0][0] = 0;
  field[0][0][0][1] = 0;

}

//========================================================================
void forcey(float pot[NCELL][NCELL][NCELL], float fx[NCELL][NCELL][NCELL]) {
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        fx[i][j][k] = (pot[i][(j + 1) & (NCELL - 1)][k] - pot[i][(j - 1)
            & (NCELL - 1)][k]) / (2. * DX);
      }
    }
  }

}

//========================================================================
void forcez(float pot[NCELL][NCELL][NCELL], float fx[NCELL][NCELL][NCELL]) {

  int i, j, k;

  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        fx[i][j][k] = (pot[i][j][(k + 1) & (NCELL - 1)] - pot[i][j][(k - 1)
            & (NCELL - 1)]) / (2. * DX);
      }
    }
  }

}

/**
 * Compute the mapping between particule position and Grid coordinate
 */
void part2data(int np,
               coord pos[NCELL][NCELL][NCELL],
               int data[NCELL][NCELL][NCELL]) {
  int i, j, k;
  float x, y, z;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        x = pos[i][j][k]._[0];
        y = pos[i][j][k]._[1];
        z = pos[i][j][k]._[2];
        data[i][j][k] = (int)(x / DX) * NCELL * NCELL + (int)(y / DX) * NCELL
            + (int)(z / DX);
      }
    }
  }

}

//========================================================================
void int2float(int v1[NCELL][NCELL][NCELL], float v2[NCELL][NCELL][NCELL]) {
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        v2[i][j][k] = v1[i][j][k] - 1; // Densité moyenne = 0
      }
    }
  }
}

//========================================================================
void correction_pot(float pot[NCELL][NCELL][NCELL],
                    float coeff[NCELL][NCELL][NCELL]) {
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        pot[i][j][k] = (float)(pot[i][j][k]) * coeff[0][0][0] / (DX * DX * DX);
      }
    }
  }
}

//========================================================================
void updatepos(coord pos[NCELL][NCELL][NCELL], coord vel[NCELL][NCELL][NCELL]) {
  float posX, posY, posZ;
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        posX = pos[i][j][k]._[0] + vel[i][j][k]._[0] * DT;
        posY = pos[i][j][k]._[1] + vel[i][j][k]._[1] * DT;
        posZ = pos[i][j][k]._[2] + vel[i][j][k]._[2] * DT;
        pos[i][j][k]._[0] = posX + LBOX * ((posX < 0) - (posX > LBOX)); // FIXME si vel * dt > LBOX la correction ne suffit pas et l'histo segfault
        pos[i][j][k]._[1] = posY + LBOX * ((posY < 0) - (posY > LBOX));
        pos[i][j][k]._[2] = posZ + LBOX * ((posZ < 0) - (posZ > LBOX));
      }
    }
  }

}

//========================================================================
void updatevel(coord vel[NCELL][NCELL][NCELL],
               float force[NCELL][NCELL][NCELL],
               int data[NCELL][NCELL][NCELL],
               int c,
               float dt) {
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        int x = floor(((float)data[i][j][k]) / (float)(NCELL * NCELL));
        int y = floor(((float)(data[i][j][k] - x * NCELL * NCELL))
            / (float)(NCELL));
        int z = data[i][j][k] - x * NCELL * NCELL - y * NCELL;
        vel[i][j][k]._[c] += force[x][y][z] * dt;
        //        vel[i][j][k][coord] += ((float *)force)[data[i][j][k]] * dt;
      }
    }
  }
}

void printUsage(int argc, char **argv) {
  printf("Usage : \n\n%s (-f||--file) file\n", argv[0]);
}

void CPUhisto(int data[NCELL][NCELL][NCELL], int histo[NCELL][NCELL][NCELL]) {
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        int x = floor(((float)data[i][j][k]) / (float)(NCELL * NCELL));
        int y = floor(((float)(data[i][j][k] - x * NCELL * NCELL))
            / (float)(NCELL));
        int z = data[i][j][k] - x * NCELL * NCELL - y * NCELL;
        ++histo[x][y][z];
        //    ++((int*)histo)[((int *))data[i]];
      }
    }
  }
}

// Initialisation des tableaux des données sur la carte graphique
//========================================================================
int main(int argc, char **argv) {
  float mp[NCELL][NCELL][NCELL]; // Mass for each particle
  coord pos[NCELL][NCELL][NCELL]; // Position (x,y,z) for each particle
  coord vel[NCELL][NCELL][NCELL]; // Velocity (x,y,z) for each particle
  // These are temporary
  float dens[NCELL][NCELL][NCELL]; // Density for each particle
  float force[NCELL][NCELL][NCELL]; // Force for each particle

  int data[NCELL][NCELL][NCELL];
  int histo[NCELL][NCELL][NCELL];

  unsigned int np = NPART;

  float cdens[NCELL][NCELL][NCELL][2];
  //  fftwf_complex *cdens; // Density, complex representation for FFT
  fftwf_plan fft_forward;
  fftwf_plan fft_backward;

  float time;

  int npdt = 0;
  float dt = DT / 2; // 1/2 only for the first iteration

  // Get initial conditions input file
  char * icfile = argv[1];
  if(NULL == icfile) {
    printUsage(argc, argv);
    exit(-1);
  }

  // Read input file
  readDataFile(np, mp, pos, vel, icfile);

  //  cdens = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * np);
  fft_forward = fftwf_plan_dft_3d(NCELL,
                                  NCELL,
                                  NCELL,
                                  (fftwf_complex*)cdens,
                                  (fftwf_complex*)cdens,
                                  FFTW_FORWARD,
                                  FFTW_ESTIMATE);
  fft_backward = fftwf_plan_dft_3d(NCELL,
                                   NCELL,
                                   NCELL,
                                   (fftwf_complex*)cdens,
                                   (fftwf_complex*)cdens,
                                   FFTW_BACKWARD,
                                   FFTW_ESTIMATE);


#ifdef _GLGRAPHICS_
      graphic_gldraw(argc, argv, pos);
#endif

  for (time = 0; time <= TMAX; time += DT) {
    if(0 == npdt % MODDISP) {
      puts("**********************************");
      printf("Time= %5.2e Npdt= %d\n", time, npdt);
    }
    //************************************  LEAPFROG  ************************************

    //************************************ NGP ************************************
    // Initialise cfg.d_data à partir des positions des particules (association particule<->cellule)
    part2data(np, pos, data);

    memset(histo, 0, np * sizeof(int));
    CPUhisto(data, histo); // construit l'histogramme (nb de particules par cellule)

#ifdef _GRAPHICS_
    if(0 == npdt % MODDISP) {
      graphic_draw(argc, argv, histo);
    }
#endif

    // conversion de format
    int2float(histo, dens);

    //************************************ Laplace Solver  ************************************

    real2Complex(cdens, dens); /* conversion de format*/
    fftwf_execute(fft_forward); /* repeat as needed */
    fft_laplacian7(cdens);
    fftwf_execute(fft_backward); /* repeat as needed */
    complex2Real(cdens, dens); // conversion de format


    correction_pot(dens, mp);

    //************************************  Finite Difference Scheme  ************************************
    forcex(dens, force);
    updatevel(vel, force, data, 0, dt);

    forcey(dens, force);
    updatevel(vel, force, data, 1, dt);

    forcez(dens, force);
    updatevel(vel, force, data, 2, dt);

    npdt++;
    //************************************  END LOOP  ********************************

    // Fix le pas de temps (1/2 DT à la première itération)
    dt = DT;

    updatepos(pos, vel);


#ifdef _GLGRAPHICS_
      graphic_glupdate(pos);
#endif

#ifdef _DUMPPOS_
      dump_pos(pos,npdt);
#endif

    //		printf("sum : %f\n",sum);/*exit(-1);		*/
  }

  // Libère les allocations
  fftwf_destroy_plan(fft_forward);
  fftwf_destroy_plan(fft_backward);

#ifdef _GRAPHICS_
  graphic_destroy();
#endif
#ifdef _GLGRAPHICS_
  graphic_gldestroy();
#endif

  puts("-----------------------------");
  puts("Finished");
  puts("-----------------------------");

  return 0;
}
