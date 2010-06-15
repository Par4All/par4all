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

typedef struct {
  s_with_pointer champ[5];
  int m;
} s_with_array_of_struct_with_pointer;

typedef struct {
  s_with_array champ[5];
  int m;
} s_with_array_of_struct_with_array;


int main()
{
  s_with_pointer a, b;
  s_with_array c, d;
  s_with_array_of_struct_with_pointer e,f;
  s_with_array_of_struct_with_array g, h;

  int i, j;

  // just to create a block for prettyprinting
  if(true) {
    
    // definition of a
    a.p = (int *) malloc(10 * sizeof(int));
    a.n = 2;
    for(i = 0; i < 10; i++)
      a.p[i] = i;
    
    // definition of b, also modifies the elments of a.p
    b = a;
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
    d = c;
    printf("\nstruct with array copy : \n");
    for(i = 0; i < 10; i++)
      {
	d.array[i] = i * 10;
	printf("c.array[%d] = %d; d.array[%d] = %d \n", 
	       i, c.array[i], i, d.array[i]);
      }
    
    // definition of e
    e.m = 4;
    for(i = 0; i < 5; i++)
      {
	e.champ[i].p =  malloc(10 * sizeof(int));
	for (j =0; j < 10; j++)
	  e.champ[i].p[j] = i+j;
      }
    
    // definition of f, also modifies e.champ[*].p[*]
    f = e;
    printf("\nstruct with array of structs with pointer copy : \n");
    for(i = 0; i < 5; i++)
      {
	for (j =0; j < 10; j++)
	  {
	    f.champ[i].p[j] = i*j;
	    printf("e.champ[%d].p[%d] = %d ; f.champ[%d].p[%d] = %d \n",
		   i,j,e.champ[i].p[j],
		   i,j,f.champ[i].p[j]);
	  }
      }
    
    // definition of g
    g.m = 5;
    for(i = 0; i < 5; i++)
      {
	for (j =0; j < 10; j++)
	  g.champ[i].array[j] = i+j;
    }
    
    // definition of h, does not modify g
    h = g;
    printf("\nstruct with array of structs with array copy : \n");
    for(i = 0; i < 5; i++)
      {
	for (j =0; j < 10; j++)
	  {
	    h.champ[i].array[j] = i*j;
	    printf("g.champ[%d].array[%d] = %d ; h.champ[%d].array[%d] = %d \n",
		   i,j,g.champ[i].array[j],
		   i,j,h.champ[i].array[j]);
	  }
      }
    
  }
  return (0);
}
