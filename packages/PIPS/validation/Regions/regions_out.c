#include <stdio.h>
#include <stdlib.h>

typedef struct coord {
   float _[3];
} coord;
int dump_pos(coord pos[128][128][128], int npdt);
int read_pos(coord pos[128][128][128], int npdt, char *fname_);


void readDataFile(unsigned int n, float mp[128][128][128], coord pos[128][128][128], coord vel[128][128][128], char *filename)
{
   unsigned int np = n;
   
   // Initialisation des particules depuis un fichier
   
   
   
   
   FILE *ic;
   ic = fopen(filename, "r");
   if (ic==(void *) 0) {
      
      // Test d'accès au fichier
      perror("Impossible d'ouvrir le fichier.");
      exit(-1);
   }
   
   // Nombre de particules dans le fichier
   if (1!=fread(&n, sizeof(int), 1, ic)) {
      perror("Erreur de lecture du nombre de particules dans le fichier.");
      fclose(ic);
      exit(-1);
   }
   // Initialise les tableaux de données
   
   if (n!=np) {
      
      
      
      fprintf(stderr, "Input file contains %d particle while we expected %d, abort\n", n, np);
      fclose(ic);
      exit(-1);
   }
   
   
   // Masses
   if (np!=fread(mp, sizeof(float), np, ic)) {
      perror("Erreur de lecture des masses dans le fichier.");
      fclose(ic);
      exit(-1);
   }

   np = 3*np;
   // Positions
   if (np!=fread(*pos, sizeof(float), np, ic)) {
      perror("Erreur de lecture des positions dans le fichier.");
      fclose(ic);
      exit(-1);
   }
   
   // Vélocité
   if (np!=fread(*vel, sizeof(float), np, ic)) {
      perror("Erreur de lecture des vélocités dans le fichier.");
      fclose(ic);
      exit(-1);
   }
   
   // Fin de lecture du fichier
   fclose(ic);
}


int dump_pos(coord pos[128][128][128], int npdt)
{
   FILE *fp;
   char fname[255];
   snprintf(fname, 255, "out%d", npdt);
   printf("dump part in %s\n", fname);

   fp = fopen(fname, "w");
   fwrite(pos, sizeof(coord), 128*128*128, fp);
   fclose(fp);

   return 0;
}


void p4a_kernel_2(int i, int j, coord pos[128][128][128], coord vel[128][128][128])
{
   //PIPS generated variable
   int k;
   //PIPS generated variable
   float posX, posY, posZ;
   // Loop nest P4A end
   if (i<=127&&j<=127)
      for(k = 0; k <= 127; k += 1) {
         posX = (pos[i][j][k]._)[0]+(vel[i][j][k]._)[0]*5*1e-2;
         posY = (pos[i][j][k]._)[1]+(vel[i][j][k]._)[1]*5*1e-2;
         posZ = (pos[i][j][k]._)[2]+(vel[i][j][k]._)[2]*5*1e-2;
         (pos[i][j][k]._)[0] = posX+6.*((posX<0)-(posX>6.));
         (pos[i][j][k]._)[1] = posY+6.*((posY<0)-(posY>6.));
         (pos[i][j][k]._)[2] = posZ+6.*((posZ<0)-(posZ>6.));
      }
}


void p4a_kernel_wrapper_2(int i, int j, coord pos[128][128][128], coord vel[128][128][128])
{
   // To be assigned to a call to P4A_vp_0: i
   // To be assigned to a call to P4A_vp_1: j
   // Loop nest P4A end
   p4a_kernel_2(i, j, pos, vel);
}


void p4a_kernel_launcher_10(coord pos[128][128][128], coord vel[128][128][128])
{
   //PIPS generated variable
   int i, j, k;
   //PIPS generated variable
   float posX, posY, posZ;
   // Loop nest P4A begin,2D(128,128)
   for(i = 0; i <= 127; i += 1)
      for(j = 0; j <= 127; j += 1)
         // To be assigned to a call to P4A_vp_0: i
         // To be assigned to a call to P4A_vp_1: j
         // Loop nest P4A end
         p4a_kernel_wrapper_2(i, j, pos, vel);
}


void updatepos(coord pos[128][128][128], coord vel[128][128][128])
{
   float posX, posY, posZ;
   int i, j, k;
   p4a_kernel_launcher_10(pos, vel);
}



int main(int argc, char **argv)
{
   float mp[128][128][128];
   // Mass for each particle
   coord pos[128][128][128];
   // Position (x,y,z) for each particle
   coord vel[128][128][128];
   // Velocity (x,y,z) for each particle
   // These are temporary
   float dens[128][128][128];
   // Density for each particle
   float force[128][128][128];
   // Force for each particle
   
   int data[128][128][128];
   int histo[128][128][128];
   unsigned int np = 128*128*128;

   float cdens[128][128][128][2];

   float time;

   int npdt = 0;
   float dt = 5*1e-2/2;

   char *icfile = argv[1];

   // Read input file
   readDataFile(np, mp, pos, vel, icfile);
   
   time = 0;


   while (time<=10) {

      updatepos(pos, vel);
      dump_pos(pos, npdt);
      time += 5*1e-2;
   }

   return 0;
}

