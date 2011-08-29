

void conditional(int cols, int jW[cols], int jE[cols]) { 
  for (int j=0; j< cols; j++) {
    jW[j] = j-1;
    jE[j] = j+1;
    if(j==0) {
      jW[0] = 0;
    }
    if(j==cols-1) {
      jE[cols-1] = cols-1;
    }
  }
}

int caller(int cols) {
  int jW[cols],jE[cols];
  
  conditional(cols,jW,jE);

  return jW[0]+jE[cols-1];

}
