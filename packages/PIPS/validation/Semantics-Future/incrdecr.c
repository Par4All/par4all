// INCREMENT - DECREMENT
// The system has 3 variables i (number of increments), d (number of decrements) and c (shared counter).
// Initially i = d = c = 0.
// There are 2 possible transitions:
// - incrementation: c++; i++;
// - decrementation: c >= 1 ? c--; d++;
// The goal is to prove the inequality i >= d always holds.
// This system was originally made to prove the language of transitions is not regular.

// $Id$

#define USE_ASSERT 0
#define RESTRUCTURE_CFG 1
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

#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#define freeze {while(1);}

// transition system

#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}

#define G (c >= 1)

#define C_incr {c++; i++;}
#define C_decr {c--; d++;}

void run(void) {
	int i, d, c;
	i = d = c = 0;
	
	assert(!G);
	check_not(BAD);
	while (1) {

#if RESTRUCTURE_CFG == 0
		
		if (flip()) {
			trans(1, C_incr);
		}
		else {
			trans(G, C_decr);
		}
		
#else
		
		while (flip()) {
			trans(c < 0, C_incr);
			assert(!G);
		}
		trans(c == 0, C_incr);
		assert(G);
		while (flip()) {
			trans(c >= 2, C_decr);
			assert(G);
		}
		trans(c == 1, C_decr);
		assert(!G);
		
#endif
		
	}
	
}

int main(void) {
	run();
	return 0;
}

