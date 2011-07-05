// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 4.7

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (20 * k > 11 * i)

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

#define S1 CONTROL(i <= 100 && j < 9)
#define S2 CONTROL(i <= 100 && j == 9)
#define S3 CONTROL(i > 100)

#define G1 (i <= 100 && j < 9)
#define G1a (i <= 98 && j < 8)
#define G1b (i <= 98 && j == 8)
#define G1c (G1 && i > 98)
#define A1 {i += 2; k++; j++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)

#define G2 (i <= 100 && j < 9)
#define G2a (i <= 98 && j < 8)
#define G2b (i <= 98 && j == 8)
#define G2c (G2 && i > 98)
#define A2 {i += 2; j++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)
#define C2c COMMAND(G2c, A2)

#define G3 (i <= 100 && j == 9)
#define G3a (i <= 98 && j == 9)
#define G3b (G3 && i > 98)
#define A3 {i += 2; k += 2; j = 0;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)

#define INI {i = j = k = 0;}

// transition system

void ts_singlestate(void) {
	int i, j, k;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int i, j, k;
	INI; CHECK;
	LOOP(
		OR(C1a; S1,
		OR(C2a; S1,
		OR(C1b; S2; C3a; S1,
		C2b; S2; C3a; S1
		)))
	)
	OR(C1c; S3,
	OR(C2c; S3,
	OR(C1b; S2; C3b; S3,
	C2b; S2; C3b; S3
	)))
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

