// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 6.8

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define GOOD (0 <= y && y <= x && y <= -x + 202 && 0 <= x && x <= 102)

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

#define G1 (s == 1 && x <= 100)
#define G1a (s == 1 && x <= 99)
#define G1b (s ==  1 && x == 100)
#define U1 {x++; y++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (s == 1 && x <= 100)
#define G2a (s == 1 && x <= 98)
#define G2b (G2 && x > 98)
#define U2 {x += 2;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define G3 (s == 1)
#define U3 {s = 2;}
#define T3 {trans(G3, U3);}

#define S1 {assert(s == 1 && x <= 100);}
#define S2 {assert(s == 1 && x > 100);}
#define S3 {assert(s == 2 && x > 100);}

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
		while (flip()) {
			T1a; S1;
		}
		while (flip()) {
			T2a; S1;
		}
	}
	if (flip()) {
		T1b; S2;
	}
	else {
		T2b; S2;
	}
	T3; S3;

#endif
	
}

int main(void) {
	run();
	return 0;
}

