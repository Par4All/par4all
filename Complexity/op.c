int main()
{
  int a,b,c,z;
  a = b + c;
  a--;
  b = a * c;
  c = b - a;
  c++;
  z = a + b * c;
  --a;
  ++z;
  z *= a;
  a /= b;
  c -= a;
  b += b;
}
