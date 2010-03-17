/* Make sure that the struct s is not expansed in the declaration of
   the anonymous union because it has been declared earlier.

   See also struct03.c and struct04.c
 */

/* definition of struct s */
struct s
{
  int l;
};

/* definition of union u_u and variable u */
union u_u
{
  struct s d;
  int i;
} u;

/* use of union u_u and definition of variable x */
//union u_u x;

/* definition of union v_v and struct s2 and variable v */
/*
union v_v
{
  struct s2 {
    int m;
  } d2;
  int i;
} v;
*/
