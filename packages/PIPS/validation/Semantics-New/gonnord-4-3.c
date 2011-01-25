// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 4.3

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define BAD (y > x || y > 1)

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

#define G1 (x >= 0)
#define U1 {x++;}
#define T1 {trans(G1, U1);}

#define G2 (x == 0)
#define U2 {x++; y++;}
#define T2 {trans(G2, U2);}

#define S1 {assert(x == 0);}
#define S2 {assert(x > 0);}

void run(void) {
	
	int x, y;
	x = y = 0;
	
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
	
	S1;
	if (flip()) {
		T1;
	}
	else {
		T2;
	}
	S2;
	while (flip()) {
		T1; S2;
	}
	
#endif
	
}

int main(void) {
	run();
	return 0;
}

