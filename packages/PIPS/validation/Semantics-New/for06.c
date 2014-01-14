// less of precision
// sum the n=10 first number

int main()
{
  int i, k;
  
  k=0;
  for (i=0; i<10; i++)
  {
    k=i+k;
  }

  // k is exactly equal to sum(0..9)=45
  return k;
}
