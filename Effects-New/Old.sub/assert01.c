#include <assert.h>

/* __assert_fail() is called by macro assert according to ISO POSIX 2003 */

/*
extern void __assert_fail (__const char *__assertion, __const char *__file,
                           unsigned int __line, __const char *__function)
     __THROW __attribute__ ((__noreturn__));
*/

void assert01()
{
  int i;

  assert(i>0);
}
