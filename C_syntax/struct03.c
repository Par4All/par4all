/* Make sure that the struct __pthread_mutex_s is fully expansed in
   the declaration of the union because it has not been declared
   earlier */

#define __WORDSIZE 64
#define __SIZEOF_PTHREAD_MUTEX_T 123
typedef double __pthread_list_t;
typedef double __pthread_slist_t;

typedef union
{
  struct __pthread_mutex_s
  {
    int __lock;
    unsigned int __count;
    int __owner;
   /* KIND must stay at this position in the structure to maintain
       binary compatibility.  */
    int __kind;
    int __spins;
    __pthread_list_t __list;
  } __data;
  char __size[__SIZEOF_PTHREAD_MUTEX_T];
  long int __align;
} pthread_mutex_t;

void main() {
}
