// This example makes PIPS r18499 segfault

#include <stdlib.h>

void run(void) {
	int n = 0;
	while (1) {
		if (rand()) {}
	}
}

int main(void) {
	run();
	return 0;
}

