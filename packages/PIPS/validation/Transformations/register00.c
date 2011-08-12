// "register" detection & forcing

void a_double(double * pd, double val)
{
  *pd = val;
}

void a_int(int * pi, int val)
{
  *pi = val;
}

typedef struct {
  int data;
} stuff_t;

void a_stuff(stuff_t * s, int val)
{
  s->data = val;
}

// special, must not be switched
int fn = 17;

// special, must not be switched
static int gn = 19;

int main(void)
{
  // scalars
  int in = 1, ir = 1;
  int kr, kn;
  auto int nr = 10;
  kr = 3;
  a_int(&in, ir);
  a_int(&kn, 17+kr);

  double jn = 2.0, jr = 2.0;
  double ln, lr;
  ln = 17.0;
  lr = 13.0;
  a_double(&ln, lr+17);
  a_double(&jn, 19.0);

  // arrays
  int an[10], ar[10];
  int bn[10][10], cn[10][10], dn[10][10], br[10][10];
  an[0] = 0;
  ar[0] = 17;
  dn[0][0] = 3;
  a_int(an, 23);
  a_int(bn+2, 27);
  a_int(&cn[3] + 5, 31);
  int sr = sizeof(ar) + sizeof(br[2]) + sizeof(dn);
  int * p = dn;
  int ** qr = &p;

  // struct
  stuff_t s1r, s1n;
  s1r.data = 10;
  s1n.data = 11;
  a_stuff(&s1n, 18);

  // special, must not be swiched
  static int sn = 3;
  sn++;

  // special, must not be switched
  extern int en;
  en = 5;

  fn = 19;
  gn++;

  // this is also okay
  void (*func)(int*,int) = &a_int;
  funcn(&in, 111);

  return 0;
}
