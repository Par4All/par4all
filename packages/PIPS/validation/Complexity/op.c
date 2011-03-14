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
  a << 1;
  b >> 1;
  a | b;
  z |= a;
  a & b;
  a ^ b;
 
  if (a!=b)
    a=b;
 
  if ((z==a || z==b) && z>1)
    z = z % c;
 
  a <<= b;
  b >>= a;
 
  if (!a)
    a=z;
  
  c ^= b;
  b &= a;
  a %= z;
  (~a) ? (z=a) : (z=b);
}
