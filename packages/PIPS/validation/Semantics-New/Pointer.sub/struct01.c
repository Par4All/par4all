struct Mastruct {
  int first;
  char second;
};

int main()
{
  struct Mastruct toto;
  int *p;
  
  p = &(toto.first);
  
  return 0;
}
