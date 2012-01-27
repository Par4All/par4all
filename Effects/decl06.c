// to make sure "a" appears only once in the proper effects
// of the declaration of "b"
int main()
{
  int a;
  a= 0;
  int b = a * a;
  return b;
}
