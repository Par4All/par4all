#include "assert.h"
#include "stdlib.h"

typedef struct t_list  {
  void* p_elem;
  struct t_list* p_next;
} t_list;

///@return a newly allocated list element
t_list* new_list_elem () {
  t_list* result = (t_list*) malloc (sizeof (t_list));
  assert(result != NULL);
  result->p_elem = NULL;
  result->p_next = NULL;
  return result;
}

///@brief append elem to the head of the list
///@return the new head of the list
///@param list, the list where to append the element
///@param the elem to append
t_list* append_int_to_list (t_list* list, int val) {
  // add the particle to the list of particles to be sent
  t_list* l =  new_list_elem();
  l->p_elem = (int*) malloc (sizeof (int));
  *((int*) (l->p_elem)) = val;
  l->p_next = list;
  return l;
}


///@brief allocate and append a value to the list
//void append_int (t_list* ptr, int* i) {
//  ptr->p_elem =  (void *) malloc (sizeof (int));
//  *(ptr->p_elem) = *i
//}

int main () {
  int val = 10;
  t_list* ptr = NULL;
  ptr = append_int_to_list (ptr, val);
  return 0;
}
