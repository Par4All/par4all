// Bertrand Jeannet: Partitionnement Dynamique dans l'Analyse de Relations Linéaires
// et Application à la Vérification de Programmes Synchrones
// figure 4.1

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (ok)

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

#define S1 CONTROL(x <= 9 && y <= 3 && ok == 1)
#define S2 CONTROL(x <= 9 && y >= 4 && ok == 1)
#define S3 CONTROL(10 <= x && x <= 13 && y <= 3 && ok == 1)
#define S4 CONTROL(10 <= x && x <= 13 && y >= 4 && ok == 1)
#define S5 CONTROL(x >= 14 && y <= 3 && ok == 1)
#define S6 CONTROL(x >= 14 && y >= 4 && ok == 1)
#define S7 CONTROL(ok == 0)

#define G1 (x >= 14 && ok == 1)
#define G1a (y <= 2 && G1)
#define G1b (y == 3 && G1)
#define A1 {x++; y++; ok = 1;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (x >= 10 && y <= 3 && ok == 1)
#define G2a (x <= 12 && y <= 2 && G2)
#define G2b (x <= 12 && y == 3 && G2)
#define G2c (x == 13 && y <= 2 && G2)
#define G2d (x == 13 && y == 3 && G2)
#define A2 A1
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)
#define C2c COMMAND(G2c, A2)
#define C2d COMMAND(G2d, A2)

#define G3 (x >= 10 && x <= 13 && y >= 4 && ok == 1)
#define A3 {x++; y++; ok = 0;}
#define C3 COMMAND(G3, A3)

#define G4 (x <= 9 && y <= 3 && ok == 1)
#define G4a (x <= 8 && G4)
#define G4b (x == 9 && G4)
#define A4 {x++; ok = 1;}
#define C4 COMMAND(G4, A4)
#define C4a COMMAND(G4a, A4)
#define C4b COMMAND(G4b, A4)

#define G5 (x <= 9 && y >= 4 && ok == 1)
#define A5 {x++; ok = 0;}
#define C5 COMMAND(G5, A5)

#define INI {x = y = 0; ok = 1;}

// transition system

void ts_singlestate(void) {
	int x, y, ok;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, OR(C3, OR(C4, C5)))))
}

void ts_restructured(void) {
	int x, y, ok;
	INI; CHECK;
	S1;
	LOOP(C4a; S1);
	C4b; S3;
	LOOP(C2a; S3);
	OR(
		C2b; S4; C3; S7,
		
		OR(
			C2d; S6,
			
			C2c; S5;
			LOOP(C1a; S5);
			C1b; S6
		);
		LOOP(C1; S6);
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

