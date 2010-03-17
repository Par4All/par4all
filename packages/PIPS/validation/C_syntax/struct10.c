/* declaration of a struct used in a module in the compilation
   unit. Variation on struct06.c and struc09.c  */

struct s
{
  int l;
};

void struct10()
{
union
{
  struct s d1;
  struct s d2;
  int i;
} u;
}
