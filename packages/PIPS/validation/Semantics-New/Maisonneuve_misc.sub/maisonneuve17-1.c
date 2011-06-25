// Inspired by maisonneuve15, with different parameters.

// $Id$

#define USE_CHECK 1
#define BAD1 (x == 2 && y == 7)
#define BAD2 (x == 5 && y == 7)

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

#if USE_CHECK == 0
#define check(e)
#define check_not(e)
#else
#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#endif

#define freeze {while(1);}

#define trans(g, c) {if (g) {c; check_not(BAD1); check_not(BAD2);} else freeze;}

// transition system

#define G1 (x >= 0 && x <= 5 && y >= 0 && y <= 5)
#define U1 {x++; y++;}
#define T1 {trans(G1, U1);}

#define G2 (x >= 4 && x <= 9 && y >= 4 && y <= 9)
#define U2 {x--; y--;}
#define T2 {trans(G2, U2);}

void run(void) {
	
	int x, y;
	x = rand();
	y = rand();
	if (G1 && y <= 5 - x) {
		
		while (1) {
			check_not(BAD1);
			check_not(BAD2);
			if (flip()) {
				T1;
			}
			else {
				T2;
			}
		}
		
	}
	
}

int main(void) {
	run();
	return 0;
}

