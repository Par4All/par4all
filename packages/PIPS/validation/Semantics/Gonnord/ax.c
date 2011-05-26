/*Source Code From Laure Gonnord*/
//Compsys code

int ax() {
  int i,j,n;

  assume(n>0);
  i=0;
  do {
    j=0;
    while(j<n-1) j++;
    i++;
  }
  while(j>=n-1 && i<n-1);
  return 0;
}

//loop exit : {i+1>=n, j>=0, j+1>=n, i>=1}
