void point_to03()
{
  typedef double a_t[10][20];
  a_t a;
  a_t * p;
  
  p = &a;
  (*p)[2][3] = 1.5;
}
