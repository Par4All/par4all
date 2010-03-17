/* Make sure s is not redeclared in struct12 declaration.

 If a semi-colon is added after the formal parameter u, the syntax is
 still OK with gcc, but incompatible with PIPS splitter syntax...

 The compilation unit is parsed and prettyprinted.
 But the PIPS parser core dumps on struct12.
 */

struct s
{
  int l;
};

void struct12(union u_u
{
  struct s d1;
  struct s d2;
  int i;
} u)
{
  int i;
}
