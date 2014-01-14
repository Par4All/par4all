#include <stdio.h>
#include <stdlib.h>


#define NF 7
#define NP 28
// ------------------------------------------------------------------------------------------------------
void sumMatrices_AOS(float** restrict fmat, float** restrict prmat, int i0, int i1, int j0, int j1)
// -------------------**---------------------------------------------------------------------------------
{
    
  int m = i1 - i0 + 1;
  int n = j1 - j0 + 1;
  int i,j,k;  
  int pos=0;
  // block #1
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
            
      pos = 0;
      for( int k1=0 ; k1<NF ; k1++ ) {
	for( int k2=k1 ; k2<NF ; k2++ ) {
                    
	  prmat[i][j*NP+pos] = fmat[i][j*NF+k1] * fmat[i][j*NF+k2];
	  pos++;
	}
      }
            
    }
  }
    
  // block #2: prolog
  for( int i=1 ; i<m ; i++ ) {
    for( int k=0 ; k<NP ; k++ ) {
      prmat[i][0*NP+k] += prmat[i-1][0*NP+k];
    }
  }
  for( int j=1; j<n ; j++ ) {
    for( int k=0 ; k<NP ; k++ ) {
      prmat[0][j*NP+k] += prmat[0][(j-1)*NP+k];
    }
  }
    
  // block #2
  for( i=1 ; i<m ; i++ ) {
    for(  j=1 ; j<n ; j++) {
      for(  k=0 ; k<NP ; k++ ) {
	prmat[i][j*NP+k] += prmat[i][(j-1)*NP+k] + prmat[i-1][j*NP+k] - prmat[i-1][(j-1)*NP+k];
      }
    }
  }
  
    
  // block #3 prolog
  for( int i=1 ; i<m ; i++ ) {
    for( int k=0 ; k<NF ; k++ ) {
      fmat[i][0*NF+k] += fmat[i-1][0*NF+k];
    }
  }
  for( int j=1 ; j<n ; j++ ) {
    for( int k=0 ; k<NF ; k++ ) {
      fmat[0][j*NF+k] += fmat[0][(j-1)*NF+k];
    }
  }


    
  // block #3
  for(  i=1 ; i<m ; i++ ) {
    for(  j=1 ; j<n ; j++) {
      for(  k=0 ; k<NF ; k++ ) {
	fmat[i][j*NF+k] += fmat[i][(j-1)*NF+k] + fmat[i-1][j*NF+k] - fmat[i-1][(j-1)*NF+k];
      }
    }
  }
  return;
}
