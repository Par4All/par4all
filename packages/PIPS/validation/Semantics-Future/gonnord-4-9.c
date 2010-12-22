// Inspired from:
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires; figure 4.9

// $Id$

#define USE_ASSERT 0
#define GOOD (0 <= i && 0 <= j + 100 && i <= j + 201)

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

#define G1 (s == 1 && i <= 100)
#define G1a (s == 1 && i < 99)
#define G1b (s == 1 && i == 100)
#define C1 {s = 2; i++;}
#define T1 {trans(G1, C1);}
#define T1a {trans(G1a, C1);}
#define T1b {trans(G1b, C1);}

#define G2 (s == 2 && j <= 19)
#define G2a (G2 && j + i <= 19)
#define G2b (G2 && j + i > 19)
#define C2 {j += i;}
#define T2 {trans(G2, C2);}
#define T2a {trans(G2a, C2);}
#define T2b {trans(G2b, C2);}

#define G3 (s == 2)
#define C3 {s = 1;}
#define T3 {trans(G3, C3);}

#define G4 (s == 1 && i > 100)
#define C4 {s = 3;}
#define T4 {trans(G4, C4);}

#define S1 {assert(s == 1 && i <= 100 && j <= 19);}
#define S2 {assert(s == 1 && i <= 100 && j > 19);}
#define S3 {assert(s == 1 && i > 100 && j <= 19);}
#define S4 {assert(s == 1 && i > 100 && j > 19);}
#define S5 {assert(s == 2 && i <= 100 && j <= 19);}
#define S6 {assert(s == 2 && i <= 100 && j > 19);}
#define S7 {assert(s == 2 && i > 100 && j <= 19);}
#define S8 {assert(s == 2 && i > 100 && j > 19);}
#define S9 {assert(s == 3);}

void run(void) {
	int s, i, j;
	s = 1;
	i = 0;
	j = -100;

	S1;
	if (flip()) {
		while (flip()) {
			T1a; S5;
			while (flip()) {
				T2a; S5;
			}
			T3; S1;
		}
		T1b; S7;
		while (flip()) {
			T2a; S7;
		}
		T3; S3;
		T4; S9
	}
	else {
		if (flip()) {
			T1a; S5;
			while (flip()) {
				if (flip()) {
					T2a; S5;
				}
				else {
					T3; S1;
					T1a; S5;
				}
			}
			T2b; S6;
			while (flip()) {
				T3; S2;
				T1a; S6;
			}
			T3; S2;
			T1b; S8;
		}
		else {
			T1b; S7;
			while (flip()) {
				T2a; S7;
			}
			T2b; S8;
		}
		T3; S4;
		T4; S9;
	}
	
}

int main(void) {
	run();
	return 0;
}

