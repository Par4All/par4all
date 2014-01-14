/* Pointer to a 2-D array hidden in structs
 *
 * This piece of code is derived from pointer21.c.
 *
 * It is only valid if i==0. It would be normal to write p[0][i][j].
 */

typedef struct s1 {int (*my_array)[5][6];} s1_t;

typedef struct s2 {s1_t my_struct;} s2_t;

int pointer22(s2_t s, int i, int j)
{
  int k;
  int (*p)[5][6] = s.my_struct.my_array;
  k = p[i][0][j];
  return k;
}
