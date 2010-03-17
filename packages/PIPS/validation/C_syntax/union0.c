/* taken from the stadard includes */
typedef union
{
  struct __pthread_mutex_s
  {
    int __lock;
    unsigned int __count;
    int __owner;





    int __kind;





    unsigned int __nusers;
    /*
    union
    {
      int __spins;
      __pthread_slist_t __list;
    };*/

  } __data;
  char __size[24];
  long int __align;
} pthread_mutex_t;
