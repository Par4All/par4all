/* Check behavior with C99 implicit variable __function__ and gcc
   __FUNCTION__ */

#define __function__ __FUNCTION__

void foo(const char * msg)
{
  char * p = msg;

  p++;
}

main()
{
  char * p = "hello";
  char a[] = "world";

  foo(__function__);
  foo(__FUNCTION__);
}
