//#include<stdlib.h>
typedef int * pointer;

// should catch that returned value is malloc'ed there
pointer alloc_pointer(int v)
{
  pointer p = malloc(sizeof(int));
  *p = v;
  return p;
}

void pointer_free(pointer p)
{
  free(p);
}

// no pointer assignment, no change in points-to
void pointer_set(pointer p, int v)
{
  *p = v;
}

// no pointer assignment, no change in points-to
void pointer_add(pointer q1, const pointer q2, const pointer q3)
{
  *q1 = (*q2) + (*q3);
}

// no pointer assignment, no change in points-to
int pointer_get(const pointer p)
{
  return *p;
}

int main(void)
{
  pointer p1, p2, p3;

  // could differentiate allocs based on call path?
  p1 = alloc_pointer(13);
  p2 = alloc_pointer(17);
  p3 = alloc_pointer(19);
  
  // no pointer assigned! can keep all points-to
  pointer_add(p1, p2, p3);


  // no pointer assigned! can keep all points-to
  pointer_set(p3, 23);

  // no pointer assigned! can keep all points-to
  pointer_add(p3, p2, p1);


  pointer_free(p1);
  pointer_free(p2);
  pointer_free(p3);
  return;
}
