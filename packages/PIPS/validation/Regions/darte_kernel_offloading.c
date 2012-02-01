#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

unsigned int T = 10;

/* Original code from Alain Darte "Kernel Offloading with Optimized Remote Accesses" */
void orig(int NI, int NJ, int c[NI+NJ], int a[NI], int b[NJ]) {
  int i,j;
  for(i=0; i<NI; i++) {
    for(j=0;j<NJ; j++) {
      c[i+j]+=a[i]*b[j];
    }
  }
}

/* The same one after skewing */
void skewed(int NI, int NJ, int c[NI+NJ], int a[NI], int b[NJ]) {
  int i,j;
  for(j=0;j<NJ; j++) {
    for(i=j; i<j+NI; i++) {
      c[i]+=a[i-j]*b[j];
    }
  }
}


/* The skewed code with two level rectangular tiling */
void tiled(int NI, int NJ, int c[NI+NJ], int a[NI], int b[NJ]) {
  int tj,ti,ii,jj;
  for(tj=NJ-1; tj>=T-1; tj-=T) {
    for(ti=tj-T+1; ti<=tj+NI+1; ti+=T) {
      if(1) {
        if(ti>tj && ti<=tj+NI-T) {
          for(ii=ti; ii<ti+T;ii++) {
            for(jj=tj;jj>tj-T;jj--) {
              c[ii]+=a[ii-jj]*b[jj];
            }
          }
        }
        if(ti==tj-T+1) {
          for(ii=ti; ii<ti+T;ii++) {
            for(jj=ii;jj>tj-T;jj--) {
              c[ii]+=a[ii-jj]*b[jj];
            }
          }
        }
        if(ti==tj+NI+1-T) {
          for(ii=ti; ii<ti+T;ii++) {
            for(jj=tj;jj>tj-T+ii-ti+1;jj--) {
              c[ii]+=a[ii-jj]*b[jj];
            }
          }
        }  
      }
    }
  }  
}

/* Check results */
#define diff(n,m,a,b) _diff(__FILE__,__LINE__,n,m,a,b)
int _diff(char *f, int l, int n, int m, int a[n+m], int b[n+m]) {
  for(int i=0; i<n; i++) {
    for(int j=0;j<m; j++) {
      if(a[i+j]!=b[i+j]) {
        printf("%s:%d ! Diff %d (%d-%d) : %d %d\n", f,l, i+j, i, j, a[i+j],b[i+j]);
        return 1;
      }
    }
  }
  return 0;
}


/* Call all version of the code and check results */
// NJ multiple de T
// NI multiple de T
int region(int NI, int NJ, int c[NI+NJ], int a[NI], int b[NJ]) {
  int tj, ti, ii, jj, i, j;
  int c_ref[NI+NJ];
  int c_orig[NI+NJ];


  for(i=0; i<NI; i++) {
    for(j=0;j<NJ; j++) {
      c[i+j]=0;
      a[i] = rand();
      b[j] = rand();
    }
  }
  
  memcpy(c_orig, c, sizeof(int)*(NI+NJ));
  
  orig(NI, NJ, c, a, b);
  memcpy(c_ref, c, sizeof(int)*(NI+NJ));
  
  diff(NJ, NI, c, c_ref);
  
  memcpy(c, c_orig, sizeof(int)*(NI+NJ));
  skewed(NI, NJ, c, a, b);

  diff(NJ, NI, c, c_ref);
  memcpy(c, c_orig, sizeof(int)*(NI+NJ));
  
  tiled(NI, NJ, c, a, b);
  diff(NJ, NI, c, c_ref);

  return 0;
}


/* Get tile size and array size from command line */
int main(int argc, char **argv) {
  if(argc>=2) {
    T = atoi(argv[1]);
  }
  int NI=2*T, NJ=4*T;
  if(argc>=3) {
    NI = atoi(argv[2]) * T;
  }
  if(argc>=4) {
    NJ = atoi(argv[3]) * T;
  }
  // C99 allocation
  int a[NI], b[NJ], c[NI+NJ];
  region(NI, NJ, c, a, b);   
}
