extern int decl33();
int decl33(int, int);
extern int decl33();

main()
{
  int k = decl33(2,3);
}

int decl33(i, j)
     int i;
     int j;
{
  return i+j;
}
