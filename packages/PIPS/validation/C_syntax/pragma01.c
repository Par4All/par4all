
void pragma01()
{
  int i;

   #  pragma omp parallel for
  for(i = 0; i < 10; i++) {
    int j = i + 1;
  }
  /* Some comment */
   #  \
 pragma omp parallel for
   // And other
  for(i = 0; i < 10; i++) {
    int j = i + 1;
  }
# pragma omp \
  parallel\
  for
  for(i = 0; i < 10; i++) {
    int j = i + 1;
  }
}
