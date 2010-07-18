// Check Memory Allocation offsets for local dynamic and global
// variables of known sizes in one C file

int global1;
int global2;

void foo()
{
  int m;
}

int main()
{
  int k;
  foo();
  return 0;
}
