// Check for const after pointers

// Constant pointer
// typedef int * const ipc_t;

// Pointer to a constant
//typedef const int * ipc_t;

// Pointer to a constant
typedef int const * ipc_t;

// Code used to avoid reading the C standard...

/*
int main()
{
  int i = 2;
  ipc_t p = &i;
  p++;
  *p = 1;
  return 0;
}
*/
