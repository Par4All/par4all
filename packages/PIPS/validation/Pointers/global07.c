/* Synthesize stubs for global pointers local to a C file */

static int i;
// int j;
// int * p_amira=&i;
// int * p=&i;
static int * p;

static int k;

int global07()
{
  int * q;

  //q = p_amira;
  q = p;

  return 0;
}
