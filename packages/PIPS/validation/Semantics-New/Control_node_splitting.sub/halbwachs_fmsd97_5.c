// David Merchat: Réduction du nombre de variables en analyse de relations
// linéaires
// figure 4.9
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 4.6

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (s >= 3)

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

#define S1 CONTROL(s == 1 && d <= 8 && v <= 1 && t <= 2)
#define S2 CONTROL(s == 1 && d <= 8 && v <= 1 && t >= 3)
#define S3 CONTROL(s == 1 && d <= 8 && v == 2 && t <= 2)
#define S4 CONTROL(s == 1 && d == 9 && v <= 1 && t <= 2)
#define S5 CONTROL(s == 1 && d == 9 && v == 2 && t <= 2)
#define S6 CONTROL(s == 1 && d <= 8 && v == 2 && t >= 3)
#define S7 CONTROL(s == 1 && d == 9 && v == 2 && t >= 3)
#define S8 CONTROL(s == 1 && d == 9 && v <= 1 && t >= 3)
#define S9 CONTROL(s == 2)

#define G1 (s == 1 && t <= 2)
#define G1a (s == 1 && t <= 1)
#define G1b (s == 1 && t == 2)
#define A1 {v = 0; t++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (s == 1 && v <= 1 && d <= 8)
#define G2a (s == 1 && v <= 0 && d <= 7)
#define G2b (s == 1 && v <= 0 && d == 8)
#define G2c (s == 1 && v == 1 && d <= 7)
#define G2d (s == 1 && v == 1 && d == 8)
#define A2 {v++; d++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)
#define C2c COMMAND(G2c, A2)
#define C2d COMMAND(G2d, A2)

#define G3 (s == 1 && t >= 3)
#define A3 {s = 2;}
#define C3 COMMAND(G3, A3)

#define G4 (s == 1 && d >= 10)
#define A4 {s = 3;}
#define C4 COMMAND(G4, A4)

#define G5 (s == 1 && v >= 3)
#define A5 {s = 4;}
#define C5 COMMAND(G5, A5)

#define INI {s = 1; d = v = t = 0;}

// transition system

void ts_singlestate(void) {
	int s, d, v, t;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, OR(C3, OR(C4, C5)))))
}

void ts_restructured(void) {
	int s, d, v, t;
	INI; CHECK;
	S1;
	LOOP(OR(C1a; S1, OR(C2a; S1, C2c; S3; C1a; S1)));
	OR(
		OR(C2c; S3; C1b; S2, C1b; S2);
		LOOP(C2a; S2);
		OR(C3; S9, OR(C2c; S6; C3; S9, OR(C2d; S7; C3; S9, C2b; S8; C3; S9))),
		
		OR(
			OR(C2b; S4, C2d; S5; C1a; S4);
			LOOP(C1a; S4);
			C1b; S8,
			
			C2d; S5; C1b; S8
		);
		C3; S9;
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

