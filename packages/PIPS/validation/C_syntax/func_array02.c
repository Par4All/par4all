/* A function is equivalent to a pointer to a function

   Here "FUNC fu=*(functions[0]);" would be as good.

   See func_array_01.c
*/

int add(int a, int b){int ab; return ab;}

typedef int (*FUNC)(int a,int b);

static FUNC functions[1] = { add };

int main()
{
  FUNC fu=(functions[0]);
  return (*fu)(1,1);
}
