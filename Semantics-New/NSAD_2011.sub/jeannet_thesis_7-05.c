// Bertrand Jeannet: Partitionnement Dynamique dans l'Analyse de Relations Linéaires
// et Application à la Vérification de Programmes Synchrones
// figure 7.5

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (b == 0 && x >= 0)

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

#define S1 CONTROL(b == 1)
#define S2 CONTROL(b == 0)

#define G1 (b == 1)
#define A1 {x++;}
#define C1 COMMAND(G1, A1)

#define G2 (b == 0)
#define A2 {x--;}
#define C2 COMMAND(G2, A2)

#define INI {\
	x = rand_z();\
	b = rand_b();\
	ASSUME(x < 0);\
}

// transition system

void ts_singlestate(void) {
	int b, x;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int b, x;
	INI; CHECK;
	if (b == 1) goto L1;
	else goto L2;
L1:
	S1;
	C1;
	goto L1;
L2:
	S2;
	C2;
	goto L2;
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

