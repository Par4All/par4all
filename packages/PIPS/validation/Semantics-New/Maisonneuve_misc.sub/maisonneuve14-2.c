// Same as maisonneuve14-1, using graph restructuration
// This time we can check that n != 0

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define BAD (n == 1)

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

#define G1 (n == 0)
#define U1 {n = 2;}
#define T1 {trans(G1, U1);}

#define G2 (n == 2)
#define U2 {n = 0;}
#define T2 {trans(G2, U2);}

#define S1 {assert(n == 0);}
#define S2 {assert(n == 2);}

void run(void) {

	int n = 0;

	while (1) {
		S1;
		T1; S2;
		T2; S1;
	}

}

int main(void) {
	run();
	return 0;
}

