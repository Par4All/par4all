// BERKELEY CACHE CONSISTENCY PROTOCOL
// Randy H. Katz, Susan J. Eggers, David A. Wood, C. L. Perkins, R. G. Sheldon:
// Implementing A Cache Consistency Protocol. ISCA 1985:276-283

// $Id$

#define USE_ASSERT 0
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

#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#define freeze {while(1);}

// transition system

#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}

#define G_i (i >= 1)
#define G_ne_uo (ne + uo >= 1)

#define C_RM {ne += e; uo++; i--; e = 0;}
#define C_WH {i += ne + uo - 1; e++; ne = 0; uo = 0;}
#define C_WM {i += e + ne + uo - 1; e = 1; ne = 0; uo = 0;}

void run(void) {
	int e, ne, uo, i;
	e = ne = uo = 0;
	i = rand();
	
	if (i >= 1) {
		check_not(BAD);
		while (1) {
			
			if (flip()) {
				trans(G_i && e + ne + uo + i >= 2, C_WM);
				assert(G_i && !G_ne_uo);
			}

			else if (flip()) {
				trans(i >= 2, C_RM);
				assert(G_i && G_ne_uo);
				while (flip()) {
					trans(i >= 2, C_RM);
					assert(G_i && G_ne_uo);
				}
				if (flip()) {
					trans(G_ne_uo, C_WH);
				}
				else {
					trans(G_i, C_WM);
				}
				assert(G_i && !G_ne_uo);
			}

			else if (flip()) {
				trans(i == 1, C_RM);
				assert(!G_i && G_ne_uo);
				trans(ne + uo >= 2, C_WH);
				assert(G_i && !G_ne_uo);
			}

			else if (flip()) {
				trans(i >= 2, C_RM);
				assert(G_i && G_ne_uo);
				while (flip()) {
					trans(i >= 2, C_RM);
					assert(G_i && G_ne_uo);
				}
				trans(i == 1, C_RM);
				assert(!G_i && G_ne_uo);
				trans(ne + uo >= 2, C_WH);
				assert(G_i && !G_ne_uo);
			}

			else if (flip()) {
				trans(i == 1 && e + ne + uo == 0, C_WM);
				assert(!G_i && !G_ne_uo);
				freeze;
			}
			
			else {
				trans(i == 1, C_RM);
				assert(!G_i && G_ne_uo);
				trans(ne + uo == 1, C_WH);
				assert(!G_i && !G_ne_uo);
				freeze;
			}
			
		}
	}

}

int main(void) {
	run();
	return 0;
}

