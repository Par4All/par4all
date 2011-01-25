// CAR SPEED CONTROLLER
// N. Halbwachs. Delay analysis in synchronous programs. In Fifth Conference on
// Computer-Aided Verification, CAV'93, Elounda (Greece), July 1993. LNCS 697,
// Springer Verlag.

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define BAD (s == 3 || s == 4)

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

#define G1 (s == 1 && t <= 2)
#define G1a (s == 1 && t <= 1)
#define G1b (s == 1 && t == 2)
#define U1 {v = 0; t++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (s == 1 && v <= 1 && d <= 8)
#define G2a (s == 1 && v <= 0 && d <= 7)
#define G2b (s == 1 && v <= 0 && d == 8)
#define G2c (s == 1 && v == 1 && d <= 7)
#define G2d (s == 1 && v == 1 && d == 8)
#define U2 {v++; d++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}
#define T2c {trans(G2c, U2);}
#define T2d {trans(G2d, U2);}

#define G3 (s == 1 && t >= 3)
#define U3 {s = 2;}
#define T3 {trans(G3, U3);}

#define G4 (s == 1 && d >= 10)
#define U4 {s = 3;}
#define T4 {trans(G4, U4);}

#define G5 (s == 1 && v >= 3)
#define U5 {s = 4;}
#define T5 {trans(G5, U5);}

#define S1 {assert(s == 1 && d <= 8 && v <= 1 && t <= 2); check_not(G4); check_not(G5);}
#define S2 {assert(s == 1 && d <= 8 && v <= 1 && t >= 3); check_not(G4); check_not(G5);}
#define S3 {assert(s == 1 && d <= 8 && v == 2 && t <= 2); check_not(G4); check_not(G5);}
#define S4 {assert(s == 1 && d == 9 && v <= 1 && t <= 2); check_not(G4); check_not(G5);}
#define S5 {assert(s == 1 && d == 9 && v == 2 && t <= 2); check_not(G4); check_not(G5);}
#define S6 {assert(s == 1 && d <= 8 && v == 2 && t >= 3); check_not(G4); check_not(G5);}
#define S7 {assert(s == 1 && d == 9 && v == 2 && t >= 3); check_not(G4); check_not(G5);}
#define S8 {assert(s == 1 && d == 9 && v <= 1 && t >= 3); check_not(G4); check_not(G5);}
#define S9 {assert(s == 2); check_not(G4); check_not(G5);}

void run(void) {
	
	int s, d, v, t;
	s = 1;
	d = v = t = 0;
	
#if RESTRUCTURE == 0
	
	while (flip()) {
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
		else {
			T5;
		}
	}
	
#else
	
	S1;
	while (flip()) {
		if (flip()) {
			T1a;
			S1;
		}
		else if (flip()) {
			T2a; S1;
		}
		else {
			T2c; S3;
			T1a; S1;
		}
	}
	if (flip()) {
		if (flip()) {
			T2c; S3;
			T1b; S2;
		}
		else {
			T1b; S2;
		}
		while (flip()) {
			T2a; S2;
		}
		if (flip()) {
			T3; S9;
		}
		else if (flip()) {
			T2c; S6;
			T3; S9;
		}
		else if (flip()) {
			T2d; S7;
			T3; S9;
		}
		else if (flip()) {
			T2b; S8;
			T3; S9;
		}
	}
	else {
		if (flip()) {
			if (flip()) {
				T2b; S4;
			}
			else {
				T2d; S5;
				T1a; S4;
			}
			while (flip()) {
				T1a; S4;
			}
			T1b; S8;
		}
		else {
			T2d; S5;
			T1b; S8;
		}
		T3; S9;
	}

#endif
	
}

int main(void) {
	run();
	return 0;
}

