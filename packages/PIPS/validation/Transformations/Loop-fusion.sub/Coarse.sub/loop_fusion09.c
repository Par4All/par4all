


void dont_fuse(int a[10][10],int b[10][10]) {
  int i;
// This fusion is forbiden if loops are parallelized !
  for(i = 0; i<10; i++) {
    a[i][9]=0;
  }
  for(i = 0; i<10; i++) {
    b[i][0]=a[1][i];
  }

}
