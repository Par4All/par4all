/* Bug encountered in SPEC2000/ammp: functions and intrinsics must
   not be named with scope information, while pointers to functions
   and functional typedefs must carry the scope information */

int main()
{
  int i;
  void foo();
  void (*bar)();
  typedef void (*barbar)();
  int fputs();
  //char line[256], *fgets() ;
  foo(i);
  (*bar)(i);
}
