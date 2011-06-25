// Randy H. Katz, Susan J. Eggers, David A. Wood, C. L. Perkins, R. G. Sheldon:
// Implementing A Cache Consistency Protocol. ISCA 1985:276-283

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (e >= 2 || (uo + ne >= 1 && e >= 1))

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
#define CHECK(c)
#define CHECK_NOT(c)
#else
void checking_error(void) {
	fprintf(stderr, "checking error");
	exit(2);
}
#define CHECK(c) {if (!(c)) checking_error();}
#define CHECK_NOT(c) {if (c) checking_error();}
#endif

#define COMMAND_NOCHECK(g, a) {ASSUME(g); a;}
#ifdef GOOD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK(GOOD);}
#else
#ifdef BAD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK_NOT(BAD);}
#endif
#endif

// control and commands

#define S1 CONTROL(i >= 1 && ne + uo < 1)
#define S2 CONTROL(i >= 1 && ne + uo >= 1)
#define S3 CONTROL(i < 1 && ne + uo >= 1)
#define S4 CONTROL(i < 1 && ne + uo < 1)

#define G_RM (i >= 1)
#define G_RM1 (i == 1)
#define G_RM2 (i >= 2)
#define A_RM {ne += e; uo++; i--; e = 0;}
#define C_RM COMMAND(G_RM, A_RM)
#define C_RM1 COMMAND(G_RM1, A_RM)
#define C_RM2 COMMAND(G_RM2, A_RM)

#define G_WH (ne + uo >= 1)
#define G_WH1 (ne + uo == 1)
#define G_WH2 (ne + uo >= 2)
#define A_WH {i += ne + uo - 1; e++; ne = 0; uo = 0;}
#define C_WH COMMAND(G_WH, A_WH)
#define C_WH1 COMMAND(G_WH1, A_WH)
#define C_WH2 COMMAND(G_WH2, A_WH)

#define G_WM (i >= 1)
#define G_WM1 (G_WM && e + ne + uo + i - 1 >= 1)
#define G_WM2 (G_WM && e + ne + uo + i - 1 < 1)
#define A_WM {i += e + ne + uo - 1; e = 1; ne = 0; uo = 0;}
#define C_WM COMMAND(G_WM, A_WM)
#define C_WM1 COMMAND(G_WM1, A_WM)
#define C_WM2 COMMAND(G_WM2, A_WM)

#define INI {\
	e = ne = uo = 0;\
	i = rand(); ASSUME(i >= 1);\
}

// transition system

void ts_singlestate(void) {
	int e, ne, uo, i;
	INI;
	LOOP(OR(C_RM, OR(C_WH, C_WM)));
}

void ts_restructured(void) {
	int e, ne, uo, i;
	INI;
	LOOP(
		S1;
		OR(
			C_WM1; S1,
		OR(
			C_RM2; S2; LOOP(C_RM2; S2); OR(C_WH; S1, C_WM; S1),
		OR(
			C_RM1; S3; C_WH2; S1,
		OR(
			C_RM2; S2; LOOP(C_RM2; S2); C_RM1; S3; C_WH2; S1,
			
			OR(
				OR(
					C_RM2; S2; LOOP(C_RM2; S2); C_RM1; S3,
					C_RM1; S3
				);
				C_WH1; S4,

				C_WM2; S4
			);
			deadlock();
		))))
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

