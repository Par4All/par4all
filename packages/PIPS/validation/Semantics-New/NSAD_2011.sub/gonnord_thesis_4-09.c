// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 4.9

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (0 <= i && 0 <= j + 100 && i <= j + 201)

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

#define S1 CONTROL(s == 1 && i <= 100 && j <= 19)
#define S2 CONTROL(s == 1 && i <= 100 && j > 19)
#define S3 CONTROL(s == 1 && i > 100 && j <= 19)
#define S4 CONTROL(s == 1 && i > 100 && j > 19)
#define S5 CONTROL(s == 2 && i <= 100 && j <= 19)
#define S6 CONTROL(s == 2 && i <= 100 && j > 19)
#define S7 CONTROL(s == 2 && i > 100 && j <= 19)
#define S8 CONTROL(s == 2 && i > 100 && j > 19)
#define S9 CONTROL(s == 3)

#define G1 (s == 1 && i <= 100)
#define G1a (s == 1 && i < 99)
#define G1b (s == 1 && i == 100)
#define A1 {s = 2; i++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (s == 2 && j <= 19)
#define G2a (G2 && j + i <= 19)
#define G2b (G2 && j + i > 19)
#define A2 {j += i;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 2)
#define A3 {s = 1;}
#define C3 COMMAND(G3, A3)

#define G4 (s == 1 && i > 100)
#define A4 {s = 3;}
#define C4 COMMAND(G4, A4)

#define INI {s = 1; i = 0; j = -100;}

// transition system

void ts_singlestate(void) {
	int s, i, j;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, OR(C3, C4))))
}

void ts_restructured(void) {
	int s, i, j;
	INI; CHECK;
	OR(
		LOOP(C1a; S5; LOOP(C2a; S5); C3; S1);
		C1b; S7;
		LOOP(C2a; S7);
		C3; S3; C4; S9,
		
		OR(
			C1a; S5;
			LOOP(OR(C2a; S5, C3; S1; C1a; S5));
			C2b; S6;
			LOOP(C3; S2; C1a; S6);
			C3; S2; C1b; S8,
			
			C1b; S7;
			LOOP(C2a; S7);
			C2b; S8
		);
		C3; S4; C4; S9
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

