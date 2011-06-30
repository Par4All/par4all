// Bertrand Jeannet: Partitionnement Dynamique dans l'Analyse de Relations Linéaires
// et Application à la Vérification de Programmes Synchrones
// figure 4.2

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

#define S1 CONTROL(b0 == 0 && b1 == 0 && x >= y && ok == 1)
#define S2 CONTROL(b0 == 0 && b1 == 1 && x >= y && ok == 1)
#define S3 CONTROL(b0 == 1 && b1 == 0 && x >= y && ok == 1)
#define S4 CONTROL(b0 == 1 && b1 == 1 && x >= y && ok == 1)
#define S5 CONTROL(x < y && ok == 1)
#define S6 CONTROL(ok == 0)

#define G1 (b0 == 1 && b1 == 1 && ok == 1 && x >= y)
#define A1 {b0 = 0; b1 = 1; x++; ok = 1;}
#define C1 COMMAND(G1, A1)

#define G2 (b0 == 1 && b1 == 0 && ok == 1 && x >= y)
#define G2a (x >= y + 1 && G2)
#define G2b (x == y && G2)
#define A2 {b0 = 1; b1 = 1; y++; ok = 1;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (b0 == 0 && b1 == 1 && ok == 1 && x >= y)
#define G3a (x >= y + 1 && G3)
#define G3b (x == y && G3)
#define A3 {b0 = 0; b1 = 0; y++; ok = 1;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)

#define G4 (b0 == 0 && b1 == 0 && ok == 1 && x >= y)
#define A4 {b0 = 1; b1 = 0; x++; ok = 1;}
#define C4 COMMAND(G4, A4)

#define G5 (ok == 1 && x < y)
#define A5 {ok = 0;}
#define C5 COMMAND(G5, A5)

#define INI {b0 = 0; b1 = 0; x = 0; y = 0; ok = 1;}

// transition system

void ts_singlestate(void) {
	int b0, b1, x, y, ok;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, OR(C3, OR(C4, C5)))))
}

void ts_restructured(void) {
	int b0, b1, x, y, ok;
	INI; CHECK;
	S1;
	LOOP(
		OR(
			C4; S3; C2b; S5; C5; S6; deadlock();,
		OR(
			C4; S3; C2a; S4; C1; S2; C3b; S5; C5; S6; deadlock();,
			
			C4; S3; C2a; S4; C1; S2; C3a; S1;
		))
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

