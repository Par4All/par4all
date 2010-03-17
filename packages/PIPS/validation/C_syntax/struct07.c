/* This code is not standard conformant because struct s is
   redeclared in the same scope */

struct s
{
  int l;
} d1;

union
{
  struct s
  {
    int l;
  } d;
  int i;
} u;
