void malloc01()
{
  int * p = malloc(10*sizeof(int));
  //int p[10];
  int i;

  for(i=0; i<10; i++) {
    p[i] = (double) i;
  }
}
