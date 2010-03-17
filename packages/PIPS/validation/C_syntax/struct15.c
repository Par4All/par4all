/* Make sure s is not redeclared in a sizeof and s_prime is
   redeclared */

struct s
{
  int l;
};

void struct15()
{
  int i = sizeof(struct s);
  int j = sizeof(struct s_prime {int i;int j;});
  struct s_prime u;
}
