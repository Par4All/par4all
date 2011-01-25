// Sébastin Bardin: Vers un Model Checking avec accélération plate des systèmes hétérogènes; figure 2.3

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define BAD (x < 0)

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

#define G1 (s == 1 && x >= 0)
#define G1a (G1 && x + 2 < y)
#define G1b (G1 && x + 2 >= y)
#define U1 {x += 2;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (s == 2)
#define U2 {x++; y++;}
#define T2 {trans(G2, U2);}

#define G3 (s == 1 && x >= y)
#define U3 {s = 2;}
#define T3 {trans(G3, U3);}

#define G4 (s = 2)
#define G4a (G4 && x - y < 0 && x - y < y)
#define G4b (G4 && x - y < 0 && x - y >= y)
#define G4c (G4 && x - y >= 0 && x - y < y)
#define G4d (G4 && x - y >= 0 && x - y >= y)
#define U4 {x -= y; s = 1;}
#define T4 {trans(G4, U4);}
#define T4a {trans(G4a, U4);}
#define T4b {trans(G4b, U4);}
#define T4c {trans(G4c, U4);}
#define T4d {trans(G4d, U4);}

#define S1 {assert(s == 1 && x < 0 && x < y);}
#define S2 {assert(s == 1 && x < 0 && x >= y);}
#define S3 {assert(s == 1 && x >= 0 && x < y);}
#define S4 {assert(s == 1 && x >= 0 && x >= y);}
#define S5 {assert(s == 2 && x >= 0 && x >= y);}
#define S6 {assert(s == 2 && x < 0 && x >= y);}

void run(void) {
	int s, x, y;
	s = 1;
	x = rand();
	y = rand();
	if (x >= 0 && y >= 0) {
		
#if RESTRUCTURE == 0
		
		while (1) {
			if (flip()) {
				T1;
			}
			else if (flip()) {
				T2;
			}
			else if (flip()) {
				T3;
			}
			else {
				T4;
			}
		}
		
#else
		
		if (x < y) {
			S3;
			while (flip()) {
				T1a; S3;
			}
			T1b;
		}
		S4;
		
		while (1) {

			if (flip()) {
				T1; S4;
			}
			
			else if (flip()) {
				T3; S5;
				while (flip()) {
					T2; S5;
				}
				T4d; S4;
			}

			else if (flip()) {
				T3; S5;
				while (flip()) {
					T2; S5;
				}
				T4b; S2;
				while (flip()) {
					T3; S6;
					while (flip()) {
						T2; S6;
					}
					T4b; S2;
				}
				T3; S6;
				while (flip()) {
					T2; S6;
				}
				T4d; S4;
			}

			else if (flip()) {
				T3; S5;
				while (flip()) {
					T2; S5;
				}
				T4c; S3;
				while (flip()) {
					T1a; S3;
				}
				T1b; S4;
			}

			else if (flip()) {
				T3; S5;
				while (flip()) {
					T2; S5;
				}
				T4b; S2;
				while (flip()) {
					T3; S6;
					while (flip()) {
						T2; S6;
					}
					T4b; S2;
				}
				T3; S6;
				while (flip()) {
					T2; S6;
				}
				T4c; S3;
				while (flip()) {
					T1a; S3;
				}
				T1b; S4;
			}

			else if (flip()) {
				T3; S5;
				while (flip()) {
					T2; S5;
				}
				T4a; S1;
				freeze;
			}

			else {
				T3; S5;
				while (flip()) {
					T2; S5;
				}
				T4b; S2;
				while (flip()) {
					T3; S6;
					while (flip()) {
						T2; S6;
					}
					T4b; S2;
				}
				T3; S6;
				while (flip()) {
					T2; S6;
				}
				T4a; S1;
				freeze;
			}
			
		}
		
#endif

	}
	
}

int main(void) {
	run();
	return 0;
}

