//#include<stdlib.h>
typedef int * pointer;

// should catch that returned value is malloc'ed there
pointer alloc_pointer(int v)
{
  
  pointer p;
  if(0 == 0)
    p = malloc(sizeof(int));
  else
    p = malloc(sizeof(int));

  *p = v;
  return p;
}



int main(void)
{
  pointer p1, p2, p3;

  // could differentiate allocs based on call path?
  p1 = alloc_pointer(13);
  p2 = alloc_pointer(17);
  p3 = alloc_pointer(19);
  

  return;
}
