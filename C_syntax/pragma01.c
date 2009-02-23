
void pragma01()
{
  int i;

#pragma omp parallel num_threads (16)
  {
    int j;

    j = i;
  }

}
