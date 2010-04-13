// Define complex number
typedef struct {
    float re;
    float im;
} Cplfloat;

void call19(float * BASE)
{
  *BASE = 0.;
}

int main(int argc, char *argv[])
{
  Cplfloat cpl[10];
  int i;

  for(i = 0; i<10; i++)
    call19(&cpl[i].re);
  return 0;
}

