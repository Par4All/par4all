// test case from SAC: call to foo with argument &b[3][5].re
// also writes b[3][5].im,  b[3][6].re, and b[3][6].im
// -> anywhere effect

typedef struct {
    float re;
    float im;
} Cplfloat;

void foo( float BASE[4])
{
  BASE[0] = 0.;
  BASE[1] = 0.;
  BASE[2] = 0.;
  BASE[3] = 0.;
}

int main()
{
  Cplfloat  b[10][10];
  foo(&b[3][5].re);
  return 0;
}
