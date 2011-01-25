// Denis Gopan, Thomas W. Reps: Guided Static Analysis. SAS 2007: 349-365

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
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

#define G1 (s == 1 && x <= 50)
#define U1 {s = 2; y++;}
#define T1 {trans(G1, U1);}

#define G2 (s == 1 && x > 50)
#define G2a (G2 && y == 0)
#define G2b (G2 && y >= 1)
#define U2 {s = 2; y--;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define G3 (s == 2 && y >= 0)
#define G3a (G3 && x <= 49)
#define G3b (G3 && x == 50)
#define U3 {s = 1; x++;}
#define T3 {trans(G3, U3);}
#define T3a {trans(G3a, U3);}
#define T3b {trans(G3b, U3);}

#define S1 {assert(s == 1 && x <= 50 && y >= 0);}
#define S2 {assert(s == 2 && x <= 50 && y >= 0);}
#define S3 {assert(s == 1 && x > 50 && y >= 0);}
#define S4 {assert(s == 2 && x > 50 && y >= 0);}
#define S5 {assert(s == 2 && x > 50 && y < 0);}

void run(void) {
	
	int s, x, y;
	s = 1;
	x = y = 0;
	
#if RESTRUCTURE == 0
	
	while (1) {
		if (flip()) {
			T1;
		}
		else if (flip()) {
			T2;
		}
		else {
			T3;
		}
	}
	
#else
	
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

#endif
	
}

int main(void) {
	run();
	return 0;
}

