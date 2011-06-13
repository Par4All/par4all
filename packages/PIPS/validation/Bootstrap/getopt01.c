// Intrinsics linked to getopt

#include "unistd.h"

#include "getopt.h"

int main(int argc, char* argv[])
{
  char *optstring = "fo";
  int i1 = getopt(argc, argv, optstring);
  struct option * longopts;
  int * longindex;
  int i2 = getopt_long(argc, argv, optstring, longopts, longindex);
  int i3 = getopt_long_only(argc, argv, optstring, longopts, longindex);
  return 0;
}
