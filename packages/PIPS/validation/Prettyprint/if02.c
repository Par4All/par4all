// Prettyprinter enhancement: preserve all braces or at least all
// braces that are needed to avoid a gcc warning about ambiguity of
// else clauses

// adapted from code t82 from HPC Project

#include <stdio.h>

int main(int argc, char* argv[])
{
  int _u_a = 3;
  int _u_b = 5;
  int _u_c = 4;
  if ((_u_a>_u_b)) {
    if (_u_b) {
      printf("a=b");
    }
  } else {
    if ((_u_a<_u_b)) {
      if ((_u_a<_u_c)) {
	printf("a<c<b");
      }
    }
  }

  return 0;
}
