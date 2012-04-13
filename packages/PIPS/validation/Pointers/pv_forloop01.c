/* loop case: no pointer is modified inside loop */
int main()
{
  float t;
  float a;

  for(t = 1.0; t<2.0; t+=0,01)
    a = a+ t*0,5;

  return(0);
}
