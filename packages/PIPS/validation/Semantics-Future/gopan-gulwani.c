// From:
// Denis Gopan, Thomas W. Reps: Guided Static Analysis. SAS 2007: 349-365
// Sumit Gulwani, Sagar Jain, Eric Koskinen: Control-flow refinement and
// progress invariants for bound analysis. PLDI 2009: 375-385

// $Id$

#define USE_ASSERT 0
#define GOOD (s == 2 || (y <= x && y <= -x + 102))

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

#define trans(g, c) {if (g) {c; check(GOOD);} else freeze;}

#define G1 (s == 1 && x <= 50)
#define C1 {s = 2; y++;}
#define T1 {trans(G1, C1);}

#define G2 (s == 1 && x > 50)
#define G2a (G2 && y == 0)
#define G2b (G2 && y >= 1)
#define C2 {s = 2; y--;}
#define T2 {trans(G2, C2);}
#define T2a {trans(G2a, C2);}
#define T2b {trans(G2b, C2);}

#define G3 (s == 2 && y >= 0)
#define G3a (G3 && x <= 49)
#define G3b (G3 && x == 50)
#define C3 {s = 1; x++;}
#define T3 {trans(G3, C3);}
#define T3a {trans(G3a, C3);}
#define T3b {trans(G3b, C3);}

#define S1 {assert(s == 1 && x <= 50 && y >= 0);}
#define S2 {assert(s == 2 && x <= 50 && y >= 0);}
#define S3 {assert(s == 1 && x > 50 && y >= 0);}
#define S4 {assert(s == 2 && x > 50 && y >= 0);}
#define S5 {assert(s == 2 && x > 50 && y < 0);}

void run(void) {
	int s, x, y;
	s = 1;
	x = y = 0;
	
	S1;
	while (flip()) {
		T1; S2;
		T3a; S1;
	}
	T1; S2;
	T3b; S3;
	while (flip()) {
		T2b; S4;
		T3; S3;
	}
	T2a; S5;
	
}

int main(void) {
	run();
	return 0;
}

