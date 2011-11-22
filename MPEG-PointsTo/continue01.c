int continue01()
{
  /* Compute transformers for continue statement used to represent
     declarations */
  int i = 2;
  int k;
  {
    int j = i;
    int i = 3;

    k = i+j;
  }
  return k;
}
