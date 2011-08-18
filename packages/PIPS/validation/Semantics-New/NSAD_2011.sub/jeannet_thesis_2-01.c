// Bertrand Jeannet: Partitionnement Dynamique dans l'Analyse de Relations Linéaires
// et Application à la Vérification de Programmes Synchrones
// figure 2.1

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (2 <= i && i <= 20 && 0 <= j && j <= 4)

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

#define S1 CONTROL(i + j <= 20 && i >= 10)
#define S2 CONTROL(i + j <= 20 && i < 10)
#define S3 CONTROL(i + j > 20)

#define G1 (i + j <= 20 && i >= 10)
#define G1a (i + j <= 16 && i >= 10)
#define G1b (i + j > 16 && i >= 10)
#define A1 {i += 4;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (i + j <= 20 && i < 10)
#define G2a (i + j <= 17 && i < 8)
#define G2b (i + j <= 17 && i >= 8)
#define G2c (i + j > 17 && G2)
#define A2 {i += 2; j++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)
#define C2c COMMAND(G2c, A2)

#define INI {i = 2; j = 0;}

// transition system

void ts_singlestate(void) {
	int i, j;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int i, j;
	INI; CHECK;
	S2; LOOP(C2a; S2);
	OR(
		C2c; S3,
		
		C2b; S1;
		LOOP(C1a; S1);
		C1b; S3
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

