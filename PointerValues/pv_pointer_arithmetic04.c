int main()
{
  int n = 10;
  int * vect = (int *) malloc (n * sizeof(int));

  for (int i = 0; i< n; i++)
    {
      *vect = i;
      vect ++;
    }
  vect -= n;
  free(vect);
}
