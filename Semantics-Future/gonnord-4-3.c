// Inspired from:
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires; figure 4.3

// $Id$

#define USE_ASSERT 0
#define BAD (y > x || y > 1)

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

#define G1 (x >= 0)
#define C1 {x++;}
#define T1 {trans(G1, C1);}

#define G2 (x == 0)
#define C2 {x++; y++;}
#define T2 {trans(G2, C2);}

#define S1 {assert(x == 0);}
#define S2 {assert(x > 0);}

void run(void) {
	int x, y;
	x = y = 0;
	
	S1;
	if (flip()) {
		T1;
	}
	else {
		T2;
	}
	S2;
	while (flip()) {
		T1; S2;
	}
	
}

int main(void) {
	run();
	return 0;
}

