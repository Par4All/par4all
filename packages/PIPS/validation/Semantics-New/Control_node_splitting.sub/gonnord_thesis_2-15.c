// Zhou Chaochen, C. A. R. Hoare, Anders P. Ravn: A Calculus of Durations. Inf.
// Process. Lett. 40(5): 269-276 (1991)
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 2.15

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (6 * l <= t + 50)

// tools

#include <stdlib.h>
#include <stdio.h>

int rand_b(void) {
	return rand() % 2;
}
int rand_z(void) {
	return rand() - rand();
}

#define OR(t1, t2) {if (rand_b()) {t1} else {t2}}
#define LOOP(t) {while (rand_b()) {t}}

void deadlock() {
	printf("deadlock\n");
	while (1);
}
#define ASSUME(c) {if (!(c)) deadlock();}

#if DO_CONTROL == 0
#define CONTROL(c)
#else
void control_error(void) {
	fprintf(stderr, "control error");
	exit(1);
}
#define CONTROL(c) {if (!(c)) control_error();}
#endif

#if DO_CHECKING == 0
#define CHECK
#else
void checking_error(void) {
	fprintf(stderr, "checking error");
	exit(2);
}
#ifdef GOOD
#define CHECK {if (!(GOOD)) checking_error();}
#else
#ifdef BAD
#define CHECK {if (BAD) checking_error();}
#endif
#endif
#endif

#define COMMAND_NOCHECK(g, a) {ASSUME(g); a;}
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK;}

// control and commands

#define L 0
#define N 1

#define S1 CONTROL(s == L && x <= 9)
#define S2 CONTROL(s == L && 10 <= x && x <= 49)
#define S3 CONTROL(s == N && x <= 9)
#define S4 CONTROL(s == N && 10 <= x && x <= 49)
#define S5 CONTROL(s == N && x >= 50)

#define G1 (s == L && x <= 9)
#define G1a (s == L && x <= 8)
#define G1b (s == L && x == 9)
#define A1 {t++; l++; x++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (s == L)
#define A2 {x = 0; s = N;}
#define C2 COMMAND(G2, A2)

#define G3 (s == N)
#define G3a (G3 && x <= 8)
#define G3b (G3 && x == 9)
#define G3c (G3 && x <= 48)
#define G3d (G3 && x == 49)
#define A3 {t++; x++;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)
#define C3c COMMAND(G3c, A3)
#define C3d COMMAND(G3d, A3)

#define G4 (s == N && x >= 50)
#define A4 {x = 0; s = L;}
#define C4 COMMAND(G4, A4)

#define INI {\
	s = 0;\
	l = t = x = 0;\
}

// transition system

void ts_singlestate(void) {
	int s, l, t, x;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, OR(C3, C4))))
}

void ts_restructured(void) {
	int s, l, t, x;
	INI; CHECK;
	S1;
	LOOP(
		LOOP(C1a; S1);
		OR(C1b; S2; C2; S3, C2; S3);
		LOOP(C3a; S3);
		C3b; S4;
		LOOP(C3c; S4);
		C3d; S5;
		LOOP(C3; S5);
		C4; S1;
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

