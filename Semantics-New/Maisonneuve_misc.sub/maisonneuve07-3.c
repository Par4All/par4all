// Rewritting of maisonneuve07-2 with restructuration
// This time, PIPS is able to check the invariant, even if x and b are
// initially unknown

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1

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

#define G1 (x >= -b)
#define G1a (x > b + 2)
#define G1b (x == b + 2)
#define U1 {x--; t++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (x <= b)
#define G2a (x < -b - 2)
#define G2b (x == -b - 2)
#define U2 {x++; t++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define G3 (1)
#define U3 {t++;}
#define T3 {trans(G3, U3);}

#define S1 {assert(-b - 1 <= x && x <= b + 1);}
#define S2 {assert(x < - b - 1);}
#define S3 {assert(x > b + 1);}

void run(void) {
	
	int x, t, b;
	x = rand(); // works with any initial value of x
	t = 0;
	b = rand(); // works with any value of b > 0
	
	if (b > 0) {
		if (x < -b - 1) {
			S2;
			while (flip()) {
				if (flip()) {
					T2a; S2;
				}
				else {
					T3; S2;
				}
			}
			T2b; S1;
		}
		else if (x > b + 1) {
			S3;
			while (flip()) {
				if (flip()) {
					T1a; S3;
				}
				else {
					T3; S3;
				}
			}
			T1b; S1;
		}
		S1;
		while (flip()) {
			check(-b - 1 <= x && x <= b + 1);
			if (flip()) {
				T1; S1;
			}
			else if (flip()) {
				T2; S1;
			}
			else {
				T3; S1;
			}
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

