// BERKELEY CACHE CONSISTENCY PROTOCOL
// Randy H. Katz, Susan J. Eggers, David A. Wood, C. L. Perkins, R. G. Sheldon:
// Implementing A Cache Consistency Protocol. ISCA 1985:276-283

// $Id$

#define USE_ASSERT 0 // requires transformer lists
#define USE_CHECK 1
#define RESTRUCTURE 1
#define BAD (e >= 2 || (uo + ne >= 1 && e >= 1))

// usefull stuff

#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void assert_error(void) {
	exit(1);
}

void checking_error(void) {
	exit(2);
}

#if USE_ASSERT == 0
#define assert(e)
#else
#define assert(e) {if (!(e)) assert_error();}
#endif

#if USE_CHECK == 0
#define check(e)
#define check_not(e)
#else
#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#endif

#define freeze {while(1);}

#ifdef GOOD
#define trans(g, c) {if (g) {c; check(GOOD);} else freeze;}
#else
#ifdef BAD
#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}
#else
#define trans(g, c) {if (g) {c;} else freeze;}
#endif
#endif

// transition system

#define G_RM (i >= 1)
#define G_RM1 (i == 1)
#define G_RM2 (i >= 2)
#define U_RM {ne += e; uo++; i--; e = 0;}
#define T_RM {trans(G_RM, U_RM);}
#define T_RM1 {trans(G_RM1, U_RM);}
#define T_RM2 {trans(G_RM2, U_RM);}

#define G_WH (ne + uo >= 1)
#define G_WH1 (ne + uo == 1)
#define G_WH2 (ne + uo >= 2)
#define U_WH {i += ne + uo - 1; e++; ne = 0; uo = 0;}
#define T_WH {trans(G_WH, U_WH);}
#define T_WH1 {trans(G_WH1, U_WH);}
#define T_WH2 {trans(G_WH2, U_WH);}

#define G_WM (i >= 1)
#define G_WM1 (G_WM && e + ne + uo + i - 1 >= 1)
#define G_WM2 (G_WM && e + ne + uo + i - 1 < 1)
#define U_WM {i += e + ne + uo - 1; e = 1; ne = 0; uo = 0;}
#define T_WM {trans(G_WM, U_WM);}
#define T_WM1 {trans(G_WM1, U_WM);}
#define T_WM2 {trans(G_WM2, U_WM);}

#define S1 {assert(i >= 1 && ne + uo < 1);}
#define S2 {assert(i >= 1 && ne + uo >= 1);}
#define S3 {assert(i < 1 && ne + uo >= 1);}
#define S4 {assert(i < 1 && ne + uo < 1);}

void run(void) {
	
	int e, ne, uo, i;
	e = ne = uo = 0;
	i = rand();
	
	if (i >= 1) {
		while (1) {

#if RESTRUCTURE == 0
			
			if (flip()) {
				T_RM;
			}
			else if (flip()) {
				T_WH;
			}
			else {
				T_WM;
			}
			
#else
			
			S1;
			if (flip()) {
				T_WM1; S1;
			}
			else if (flip()) {
				T_RM2; S2;
				while (flip()) {
					T_RM2; S2;
				}
				if (flip()) {
					T_WH; S1;
				}
				else {
					T_WM; S1;
				}
			}
			else if (flip()) {
				T_RM1; S3;
				T_WH2; S1;
			}
			else if (flip()) {
				T_RM2; S2;
				while (flip()) {
					T_RM2; S2;
				}
				T_RM1; S3;
				T_WH2; S1;
			}
			else {
				if (flip()) {
					if (flip()) {
						T_RM2; S2;
						while (flip()) {
							T_RM2; S2;
						}
						T_RM1; S3;
					}
					else {
						T_RM1; S3;
					}
					T_WH1; S4;
				}
				else {
					T_WM2; S4;
				}
				freeze;
			}

#endif
		
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

