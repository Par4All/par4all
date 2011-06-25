// If run with PIPS r19837, this example requires transformer lists
// to be properly analyzed.

// $Ids$

#include <stdlib.h>

void error() {
	exit(1);
}

void run() {
	int n = 0;
	while (rand()) {
		if (rand() % 2) {
			n = 0;
		}
	}
	if (n == 42) error();
}

int main(void) {
  run();
  return 0;
}

