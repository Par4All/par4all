
#define NF 7
#define NP 28

// ---------------------------------------------------------------------------------------------------
void sumMatrices_SOA(float***   fmat, float***   prmat, int i0, int i1, int j0, int j1)
// ---------------------------------------------------------------------------------------------------
{
  // 1 = 0 + restrict
    
  int m = i1 - i0 + 1;
  int n = j1 - j0 + 1;

  // loop #1
  int pos = 0;
  for( int k1=0 ; k1<NF ; k1++ ) {
    for( int k2=k1 ; k2<NF ; k2++ ) {
      for(int i=0;i<m;i++) {
	for(int j=0;j<n;j++) {
	  prmat[pos][i][j] = fmat[k1][i][j] * fmat[k2][i][j];
	}
      }
      pos++;
    }
  }
    
  // loop #2
  for( int k=0 ; k<NP ; k++ ) {
        
    for( int i=1 ; i<m ; i++ )
      prmat[k][i][0] += prmat[k][i-1][0];
        
    for( int j=1; j<n ; j++ )
      prmat[k][0][j] += prmat[k][0][j-1];
        
    for( int i=1 ; i<m ; i++ ) {
      for( int j=1 ; j<n ; j++) {
	prmat[k][i][j] += prmat[k][i][j-1] + prmat[k][i-1][j] - prmat[k][i-1][j-1];
      }
    }
  }
    
  // loop #3
  for( int k=0 ; k<NF ; k++ ) {
        
    for( int i=1 ; i<m ; i++ )
      fmat[k][i][0] += fmat[k][i-1][0];
        
    for( int j=1 ; j<n ; j++ )
      fmat[k][0][j] += fmat[k][0][j-1];
        
    for( int i=1 ; i<m ; i++ ) {
      for( int j=1 ; j<n ; j++ ) {
	fmat[k][i][j] += fmat[k][i][j-1] + fmat[k][i-1][j] - fmat[k][i-1][j-1];
      }
    }
  }
  return;
}
