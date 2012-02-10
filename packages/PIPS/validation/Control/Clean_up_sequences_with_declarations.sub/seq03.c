// internal sequence entity has same same as external sequence entity
int main()
{
  int i=0,j;
  {
    int i=1;
    j = i;
  }
  return i+j;
}
