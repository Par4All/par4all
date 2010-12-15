// Inspired from:
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires; figure 2.12

// $Id$

#define USE_ASSERT 0
#define BAD (i < 2 * j || j < 0)

// usefull stuff

#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void assert_error(void) {
	exit(1);
}

void checking_error(void) {
	exit(2);
}

#if USE_ASSERT == 0
#define assert(e)
#else
#define assert(e) {if (!(e)) assert_error();}
#endif

#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#define freeze {while(1);}

// transition system

#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}

#define G (i <= 100)

#define C1 {i += 4;}
#define C2 {i += 2; j++;}

void run(void) {
	int i, j;
	i = j = 0;
	
	assert(i <= 100);
	while (flip()) {
		while (flip()) {
			trans(i <= 96, C1);
			assert(i <= 100);
		}
		while (flip()) {
			trans(i <= 98, C2);
			assert(i <= 100);
		}
	}
	if (flip()) {
		trans(96 < i && G, C1);
	}
	else {
		trans(98 < i && G, C2);
	}
	assert(i > 100);
	
}

int main(void) {
	run();
	return 0;
}

