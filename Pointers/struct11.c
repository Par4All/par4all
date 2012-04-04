/* Check initialization of struct and typedef struct
 */

struct foo {
  int one[10];
  struct foo * two;
};

typedef struct foo my_struct;

int struct11()
{
  struct foo s1;
  my_struct s2;

  return 0;
}
