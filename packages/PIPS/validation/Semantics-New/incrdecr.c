// INCREMENT - DECREMENT
// The system has 3 variables i (number of incrementations), d (number of decrementations) and c (shared counter).
// Initially i = d = c = 0.
// There are 2 possible transitions:
// - incrementation: c++; i++;
// - decrementation: c >= 1 ? c--; d++;
// The goal is to prove the inequality i >= d always holds.
// This system was originally made to prove the language of transitions is not regular.

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define BAD (i < d)

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

#define G_decr (c >= 1)
#define G_decr1 (c >= 2)
#define G_decr2 (c == 1)
#define U_decr {c--; d++;}
#define T_decr {trans(G_decr, U_decr);}
#define T_decr1 {trans(G_decr1, U_decr);}
#define T_decr2 {trans(G_decr2, U_decr);}

#define G_incr (1)
#define G_incr1 (c < 0)
#define G_incr2 (c == 0)
#define U_incr {c++; i++;}
#define T_incr {trans(G_incr, U_incr);}
#define T_incr1 {trans(G_incr1, U_incr);}
#define T_incr2 {trans(G_incr2, U_incr);}

#define S1 {assert(c < 1);}
#define S2 {assert(c >= 1);}

void run(void) {
	
	int i, d, c;
	i = d = c = 0;
	
	while (1) {

#if RESTRUCTURE == 0
		
		if (flip()) {
			T_incr;
		}
		else {
			T_decr;
		}
		
#else
		
		S1;
		while (flip()) {
			T_incr1; S1;
		}
		T_incr2; S2;
		while (flip()) {
			T_decr1; S2;
		}
		T_decr2; S1;
		
#endif
		
	}
	
}

int main(void) {
	run();
	return 0;
}

