// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 2.12

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define BAD (i < 2 * j || j < 0)

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

#define G1 (i <= 100)
#define G1a (i <= 96)
#define G1b (G1 && i > 96)
#define U1 {i += 4;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (i <= 100)
#define G2a (i <= 98)
#define G2b (G2 && i > 98)
#define U2 {i += 2; j++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define S1 {assert(i <= 100);}
#define S2 {assert(i > 100);}

void run(void) {

	int i, j;
	i = j = 0;

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
	while (flip()) {
		if (flip()) {
			T1a; S1;
		}
		else {
			T2a; S1;
		}
	}
	if (flip()) {
		T1b; S2;
	}
	else {
		T2b; S2;
	}

#endif
	
}

int main(void) {
	run();
	return 0;
}

