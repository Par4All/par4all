#define BAR "This is the bar.c file"

typedef struct bar {
  int bar_first;
  int bar_second;
} t_bar;

int bar_e(int k)
{
  return(k+5);
}

int bar_d(int k)
{
  return(k+4);
}

static int bar_c(int k)
{
  return(k+3);
}

int bar_b(int k)
{
  return(k+2);
}
