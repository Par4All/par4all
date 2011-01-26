// Sumit Gulwani: SPEED: Symbolic Complexity Bound Analysis. CAV 2009:51-62
// example 1

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define GOOD (i <= 100 + m)

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

#define G1 (x < 100 && y < m)
#define G1a (x < 100 && y < m - 1)
#define G1b (x < 100 && y == m - 1)
#define U1 {y++; i++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (x < 100 && y >= m)
#define G2a (x < 99 && y >= m)
#define G2b (x == 99 && y >= m)
#define U2 {x++; i++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define S1 {assert(x < 100 && y < m);}
#define S2 {assert(x < 100 && y >= m);}
#define S3 {assert(x >= 100 && y >= m);}

void run(void) {
	
	int x, y, m, i;
	x = y = i = 0;
	m = rand();

	if (m >= 0) {

#if RESTRUCTURE == 0

		while (1) {
			if (flip()) {
				T1;
			}
			else {
				T2;
			}
		}
		
#else
		
		if (y < m) {
			S1;
			while (flip()) {
				T1a; S1;
			}
			T1b;
		}
		
		S2;
		while (flip()) {
			T2a; S2;
		}
		T2b;
		
		S3;
		
#endif
		
	}
	
}

int main(void) {
	run();
	return 0;
}

