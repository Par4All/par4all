// Sébastin Bardin: Vers un Model Checking avec accélération plate des systèmes hétérogènes
// figure 2.3
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 3.2

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (x < 0)

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

#define S1 CONTROL(s == 1 && x < 0 && x < y)
#define S2 CONTROL(s == 1 && x < 0 && x >= y)
#define S3 CONTROL(s == 1 && x >= 0 && x < y)
#define S4 CONTROL(s == 1 && x >= 0 && x >= y)
#define S5 CONTROL(s == 2 && x >= 0 && x >= y)
#define S6 CONTROL(s == 2 && x < 0 && x >= y)

#define G1 (s == 1 && x >= 0)
#define G1a (G1 && x + 2 < y)
#define G1b (G1 && x + 2 >= y)
#define A1 {x += 2;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (s == 2)
#define A2 {x++; y++;}
#define C2 COMMAND(G2, A2)

#define G3 (s == 1 && x >= y)
#define A3 {s = 2;}
#define C3 COMMAND(G3, A3)

#define G4 (s = 2)
#define G4a (G4 && x - y < 0 && x - y < y)
#define G4b (G4 && x - y < 0 && x - y >= y)
#define G4c (G4 && x - y >= 0 && x - y < y)
#define G4d (G4 && x - y >= 0 && x - y >= y)
#define A4 {x -= y; s = 1;}
#define C4 COMMAND(G4, A4)
#define C4a COMMAND(G4a, A4)
#define C4b COMMAND(G4b, A4)
#define C4c COMMAND(G4c, A4)
#define C4d COMMAND(G4d, A4)

#define INI {\
	s = 1;\
	x = rand();\
	y = rand();\
}

// transition system

void ts_singlestate(void) {
	int s, x, y;
	INI; CHECK;
	LOOP(
		OR(C1,
		OR(C2,
		OR(C3,
		C4))))
}
		
void ts_restructured(void) {
	int s, x, y;
	INI; CHECK;
	if (x < y) {
		S3;
		LOOP(C1a; S3);
		C1b;
	}
	S4;
	LOOP(
		OR(
			C1; S4,
		OR(
			C3; S5; LOOP(C2; S5); C4d; S4,
		OR(
			C3; S5; LOOP(C2; S5); C4b; S2;
			LOOP(C3; S6; LOOP(C2; S6); C4b; S2);
			C3; S6; LOOP(C2; S6); C4d; S4,
		OR(
			C3; S5; LOOP(C2; S5); C4c; S3; LOOP(C1a; S3); C1b; S4,
		OR(
			C3; S5; LOOP(C2; S5); C4b; S2;
			LOOP(C3; S6; LOOP(C2; S6); C4b; S2);
			C3; S6; LOOP(C2; S6); C4c; S3; LOOP(C1a; S3); C1b; S4,
		OR(
			C3; S5; LOOP(C2; S5); C4a; S1; deadlock();,

			C3; S5; LOOP(C2; S5); C4b; S2;
			LOOP(C3; S6; LOOP(C2; S6); C4b; S2);
			C3; S6; LOOP(C2; S6); C4a; S1; deadlock();
		))))))
	)
	
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

