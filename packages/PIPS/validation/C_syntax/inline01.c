static inline int inline01(int x, int a)
{
  return (a > 0) ? ((x + (1 << (a-1) )) >> a) : (x << (-a));
}
