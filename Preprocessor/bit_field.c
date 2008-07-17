int main()
{
  struct x
  {
    int a:1;
    signed int b:2;
    unsigned c:3;
    long int d:4;
    char e:5;
    signed :6;
  } y;

  return 0;
}
