#include "stdbool.h"

typedef struct {
  // if bool is replaced by int the bug disapear
  bool d_is_first;
} t_elem1;

typedef struct {
  // if bool is replaced by int the bug disapear
  int d_is_first;
} t_elem2;

int main () {
  t_elem1 elem1;
  t_elem2 elem2;
  // this is ok
  if (elem2.d_is_first) {
	return 2;
  }
  // this fails
  if (elem1.d_is_first) {
	return 1;
  }
  return 0;
}
