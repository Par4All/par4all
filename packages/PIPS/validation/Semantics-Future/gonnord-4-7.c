// Inspired from:
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires; figure 4.7

// $Id$

#define USE_ASSERT 0
#define BAD (20 * k > 11 * i)

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

#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}

#define G1 (i <= 100 && j < 9)
#define G1a (i <= 98 && j < 8)
#define G1b (i <= 98 && j == 8)
#define G1c (G1 && i > 98)
#define C1 {i += 2; k++; j++;}
#define T1 {trans(G1, C1);}
#define T1a {trans(G1a, C1);}
#define T1b {trans(G1b, C1);}
#define T1c {trans(G1c, C1);}

#define G2 (i <= 100 && j < 9)
#define G2a (i <= 98 && j < 8)
#define G2b (i <= 98 && j == 8)
#define G2c (G2 && i > 98)
#define C2 {i += 2; j++;}
#define T2 {trans(G2, C2);}
#define T2a {trans(G2a, C2);}
#define T2b {trans(G2b, C2);}
#define T2c {trans(G2c, C2);}

#define G3 (i <= 100 && j == 9)
#define G3a (i <= 98 && j == 9)
#define G3b (G3 && i > 98)
#define C3 {i += 2; k += 2; j = 0;}
#define T3 {trans(G3, C3);}
#define T3a {trans(G3a, C3);}
#define T3b {trans(G3b, C3);}

#define S1 {assert(i <= 100 && j < 9);}
#define S2 {assert(i <= 100 && j == 9);}
#define S3 {assert(i > 100);}

void run(void) {
	int i, j, k;
	i = j = k = 0;
	
	S1;
	while (flip()) {
		if (flip()) {
			T1a; S1;
		}
		else if (flip()) {
			T2a; S1;
		}
		else if (flip()) {
			T1b; S2;
			T3a; S1;
		}
		else {
			T2b; S2;
			T3; S1;
		}
	}
	if (flip()) {
		T1c; S3;
	}
	else if (flip()) {
		T2c; S3;
	}
	else if (flip()) {
		T1b; S2;
		T3b; S3;
	}
	else {
		T2b; S2;
		T3b; S3;
	}
	
}

int main(void) {
	run();
	return 0;
}

