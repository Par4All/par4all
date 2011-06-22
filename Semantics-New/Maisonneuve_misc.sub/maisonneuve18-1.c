// The system contains 2 variables x, y and two transitions
// T1: y <= 2 ? x' = x+1
// T2: x <= 2 ? y' = y + 1
// Initially, x = y = 0.
// (Same example as gonnord-6-2, with different parameters.)
// 
// Reachable states are
// S = {(x, y) | x >= 0, y >= 0, (x <= 2 \/ y <= 2)}
// which is not convex. Any state (x, y) such as (x >= 3 /\ y >= 3) does not
// belong to S, thus it belongs to its convex hull
// Sc = {(x, y) | x >= 0, y >= 0}.
// 
// This file is a direct encoding of this system. In this case, PIPS is not
// able to check the property, it finds Sc as main loop invariant.

// $Id$

#define USE_CHECK 1
#define BAD (x >= 3 && y >= 3)

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

#define G1 (y <= 2)
#define U1 {x++;}
#define T1 {trans(G1, U1);}

#define G2 (x <= 2)
#define U2 {y++;}
#define T2 {trans(G2, U2);}

void run(void) {
	
	int x, y;
	x = y = 0;
	
	while (1) {
		check_not(BAD);
		if (flip()) {
			T1;
		}
		else {
			T2;
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

