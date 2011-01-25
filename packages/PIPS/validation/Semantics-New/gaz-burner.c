// LEAKING GAZ BURNER
// Zhou Chaochen, C. A. R. Hoare, Anders P. Ravn: A Calculus of Durations. Inf.
// Process. Lett. 40(5): 269-276 (1991)

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define BAD (6 * l > t + 50)

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

#define L 0
#define N 1

#define G1 (s == L && x <= 9)
#define G1a (s == L && x <= 8)
#define G1b (s == L && x == 9)
#define U1 {t++; l++; x++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (s == L)
#define U2 {x = 0; s = N;}
#define T2 {trans(G2, U2);}

#define G3 (s == N)
#define G3a (G3 && x <= 8)
#define G3b (G3 && x == 9)
#define G3c (G3 && x <= 48)
#define G3d (G3 && x == 49)
#define U3 {t++; x++;}
#define T3 {trans(G3, U3);}
#define T3a {trans(G3a, U3);}
#define T3b {trans(G3b, U3);}
#define T3c {trans(G3c, U3);}
#define T3d {trans(G3d, U3);}

#define G4 (s == N && x >= 50)
#define U4 {x = 0; s = L;}
#define T4 {trans(G4, U4);}

#define S1 {assert(s == L && x <= 9);}
#define S2 {assert(s == L && 10 <= x && x <= 49);}
#define S3 {assert(s == N && x <= 9);}
#define S4 {assert(s == N && 10 <= x && x <= 49);}
#define S5 {assert(s == N && x >= 50);}

void run(void) {
	
	int s;
	int l, t, x;
	s = 0;
	l = t = x = 0;

	while (1) {
	
#if RESTRUCTURE == 0
	
		if (flip()) {
			T1;
		}
		else if (flip()) {
			T2;
		}
		else if (flip()) {
			T3;
		}
		else if (flip()) {
			T4;
		}
		
#else
	
		S1;
		while (flip()) {
			T1a; S1;
		}
		if (flip()) {
			T1b; S2;
			T2; S3;
		}
		else {
			T2; S3;
		}
		while (flip()) {
			T3a; S3;
		}
		T3b; S4;
		while (flip()) {
			T3c; S4;
		}
		T3d; S5;
		while (flip()) {
			T3; S5;
		}
		T4; S1;
		
#endif
		
	}
	
}

int main(void) {
	run();
	return 0;
}

