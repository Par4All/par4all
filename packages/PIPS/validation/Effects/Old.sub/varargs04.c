#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>
#define MAXARGS 31


void f2(int n_ptrs, char *array[MAXARGS])
{
  int ptr_no;
  for(ptr_no=0; ptr_no < n_ptrs; ptr_no++)
    fprintf(stderr, "%s\n", array[ptr_no]);
   
}

void f4(int n_ptrs, char *array[MAXARGS])
{
  int ptr_no;
  for(ptr_no=0; ptr_no < n_ptrs; ptr_no++)
    fprintf(stdout, "%s\n", array[ptr_no]); 
}



/* adapted from ISO/IEC 9899:1999 */
void f3(int n_ptrs, int f4_after, ...)
{
      va_list ap, ap_save;
      char *array[MAXARGS];
      int ptr_no = 0;
      if (n_ptrs > MAXARGS)
            n_ptrs = MAXARGS;
      va_start(ap, f4_after);
      for(ptr_no=0; ptr_no < n_ptrs; ptr_no++) {
	array[ptr_no] = va_arg(ap, char *);
	if (ptr_no == f4_after)
	  va_copy(ap_save, ap);
      }
      va_end(ap);
      f2(n_ptrs, array);
      // Now process the saved copy.
      n_ptrs -= f4_after;
      for(ptr_no=0; ptr_no < n_ptrs; ptr_no++)
	array[ptr_no] = va_arg(ap_save, char *);
      va_end(ap_save);
      f4(n_ptrs, array);
}

int main()
{
  f3(6, 4, "ch1", "ch2", "ch3", "ch4", "ch5", "ch6");
  return 0;
}
