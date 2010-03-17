void malloc02()
{
  int *p[10];
  int i;

  for(i=0; i<10; i++) {
    p[i] = malloc(sizeof(int));
    p[i] = i;
  }
}
