// to test initialization of structs in declarations
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct {
  int * p;
  int n;
} s_with_pointer;

typedef struct {
  int array[10];
  int n;
} s_with_array;


s_with_pointer foo1(s_with_pointer s)
{
  s_with_pointer s_loc = s;
  return s_loc;
}

s_with_array foo2(s_with_array s)
{
  s_with_array s_loc = s;
  return s_loc;
}

int main()
{
  s_with_pointer a, b;
  s_with_array c, d;

  int i, j;

  // just to create a block for prettyprinting
  if(true) {
    
    // definition of a
    a.p = (int *) malloc(10 * sizeof(int));
    a.n = 2;
    for(i = 0; i < 10; i++)
      a.p[i] = i;
    
    // definition of b, also modifies the elements of a.p
    b = foo1(a);
    printf("\nstruct with pointer copy : \n");
    for(i = 0; i < 10; i++)
      {
	b.p[i] = i * 10;
	printf("a.p[%d] = %d; b.p[%d] = %d \n", i, a.p[i], i, b.p[i]);
      }
    
    // definition of c
    c.n = 3;
    for(i = 0; i < 10; i++)
      c.array[i] = i;
    
    // definition of d, does not modify c
    d = foo2(c);
    printf("\nstruct with array copy : \n");
    for(i = 0; i < 10; i++)
      {
	d.array[i] = i * 10;
	printf("c.array[%d] = %d; d.array[%d] = %d \n", 
	       i, c.array[i], i, d.array[i]);
      }
    
  }
  return (0);
}
