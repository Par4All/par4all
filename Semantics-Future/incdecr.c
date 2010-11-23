/*#define USE_VARIANT*/

#include <stdlib.h>
#include <stdio.h>

void error() {
	exit(1);
}

int alea() {
	return rand() % 2;
}

void run() {
	int s = 0, i = 0;
	int c1 = 0, c2 = 0; // compteurs pour la correction
#ifndef USE_VARIANT
	while (alea()) {
		i++;
		c1++;
	}
	s = 1;
	while (i >= 1) {
		i--;
		c2++;
	}
#else
	if (alea()) {
		i++;
		c1++;
		while (alea()) {
			i++;
			c1++;
		}
		s = 1;
		while (i >= 2) {
			i--;
			c2++;
		}
		i--;
		c2++;
	}
	else {
		s = 1;
	}
#endif
	if (c1 != c2) error();
	printf("i = %d\n", i);
}

int main(void) {
	run();
	return 0;
}

