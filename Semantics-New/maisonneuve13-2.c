// Same as maisonneuve13-1, using graph restructuration
// Transformer lists are not needed here

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define GOOD (x >= 0)

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

#ifdef GOOD
#define trans(g, c) {if (g) {c; check(GOOD);} else freeze;}
#else
#ifdef BAD
#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}
#else
#define trans(g, c) {if (g) {c;} else freeze;}
#endif
#endif

// transition system

#define G1 (x > 0)
#define G1a (x == 1)
#define G1b (x > 1)
#define U1 {x--;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (x <= 0)
#define G2a (x <= -1)
#define G2b (x == 0)
#define U2 {x++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define S1 {assert(G1);}
#define S2 {assert(G2);}

void run(void) {
	
	int x;
	x = 1 + rand();

	while (1) {
		S1;
		if (flip()) {
			T1b; S1;
		}
		else {
			T1a; S2;
			while (flip()) {
				T2a; S2;
			}
			T2b; S1;
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

