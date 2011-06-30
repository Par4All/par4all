// Bertrand Jeannet: Partitionnement Dynamique dans l'Analyse de Relations Linéaires
// et Application à la Vérification de Programmes Synchrones
// figure 8.5

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD ((s == I && x >= 11) || (s == D && x <= 4) || (s == D && x >= 12))

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

#define I 1
#define D 2

#define S1 CONTROL(s == I && x <= 5)
#define S2 CONTROL(s == I && 6 <= x && x <= 9)
#define S3 CONTROL(s == I && x >= 10)
#define S4 CONTROL(s == D && x <= 5)
#define S5 CONTROL(s == D && 6 <= x && x <= 9)
#define S6 CONTROL(s == D && x >= 10)

#define G1 (s == I && x <= 9)
#define G1a (G1 && x <= 4)
#define G1b (G1 && x == 5)
#define G1c (G1 && x <= 8)
#define G1d (G1 && x == 9)
#define A1 {x++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)
#define C1d COMMAND(G1d, A1)

#define G2 (s == I && x >= 10)
#define A2 {s = D; x++;}
#define C2 COMMAND(G2, A2)

#define G3 (s == D && x <= 5)
#define A3 {s = I; x--;}
#define C3 COMMAND(G3, A3)

#define G4 (s == D && x >= 6)
#define G4a (G4 && x == 6)
#define G4b (G4 && x >= 7)
#define G4c (G4 && x == 10)
#define G4d (G4 && x >= 11)
#define A4 {x--;}
#define C4 COMMAND(G4, A4)
#define C4a COMMAND(G4a, A4)
#define C4b COMMAND(G4b, A4)
#define C4c COMMAND(G4c, A4)
#define C4d COMMAND(G4d, A4)

#define INI {\
	s = I;\
	x = 0;\
}

// transition system

void ts_singlestate(void) {
	int s, x;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, OR(C3, C4))))
}

void ts_restructured(void) {
	int s, x;
	INI; CHECK;
	LOOP(
		S1;
		LOOP(C1a; S1);
		C1b; S2;
		LOOP(C1c; S2);
		C1d; S3;
		C2; S6;
		LOOP(C4d; S6);
		C4c; S5;
		LOOP(C4b; S5);
		C4a; S4;
		C3; S1;
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

