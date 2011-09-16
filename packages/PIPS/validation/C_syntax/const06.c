// Check for const before and after pointers

typedef int * const ipc_t;

int main()
{
  register ipc_t p;
  register int * const q;
  return 0;
}
