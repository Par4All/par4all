struct Mastruct {
  int first;
  char second;
};

int main()
{
  struct Mastruct toto;
  struct Mastruct *p;
  p = &toto;
  toto.first = 0;
  p->first = 1;
  
  return 0;
}
