/* This code is not standard conformant because struct s is
   redeclared in the same scope */

union
{
  struct s
  {
    int l;
  } d1;
  struct s
  {
    int l;
  } d2;
  int i;
} u;
