 /* package arithmetique */

 /*LINTLIBRARY*/

#include <stdio.h>

#include "arithmetique.h"
#include "assert.h"

/* int pgcd_slow(int a, int b): calcul recursif du pgcd de deux entiers;
 * le pgcd retourne est toujours positif; par extension, pgcd_slow(0,0)==1;
 * il vaudrait peut-etre mieux retourner 0 dans l'espoir de provoquer une
 * exception ulterieure par devision par 0; ou aborter immediatement;
 */
Value pgcd_slow(Value a, Value b)
{
    Value m;

    if (VALUE_ZERO_P(a) && VALUE_ZERO_P(b))
	return VALUE_ONE;

    if (VALUE_ZERO_P(a))
	return value_abs(b);

    if (VALUE_ZERO_P(b))
	return value_abs(a);

    a = value_abs(a);
    b = value_abs(b);
    m = value_mod(a,b);

    if (value_ge(a,b)) { 
	if (VALUE_ZERO_P(m))
	    return value_abs(b);
	else 
	    return pgcd(b,m);
    }
    else {
	if (VALUE_ZERO_P(m))
	    return value_abs(a);
	else
	    return pgcd(a,m);
    }
}

/* int pgcd_fast(int a, int b): calcul iteratif du pgcd de deux entiers;
 * le pgcd retourne est toujours positif; il n'est pas defini si
 * a et b sont nuls (abort);
 */
Value pgcd_fast(Value a, Value b)
{
    Value gcd;
   
    assert(VALUE_NOTZERO_P(b)||VALUE_NOTZERO_P(a));

    a = value_abs(a);
    b = value_abs(b);

    /* si cette routine n'est JAMAIS appelee avec des arguments nuls,
       il faudrait supprimer les deux tests d'egalite a 0; ca devrait
       etre le cas avec les vecteurs creux */
    if(value_gt(a,b))
	gcd = VALUE_ZERO_P(b)? a : pgcd_interne(a,b);
    else
	gcd = VALUE_ZERO_P(a)? b : pgcd_interne(b,a);

    return gcd;
}

/* int pgcd_interne(int a, int b): calcul iteratif du pgcd de deux entiers
 * strictement positifs tels que a > b;
 * le pgcd retourne est toujours positif;
 */
Value pgcd_interne(Value a, Value b)
{
    /* Definition d'une look-up table pour les valeurs de a appartenant
       a [0..GCD_MAX_A] (en fait [-GCD_MAX_A..GCD_MAX_A])
       et pour les valeurs de b appartenant a [1..GCD_MAX_B]
       (en fait [-GCD_MAX_B..GCD_MAX_B] a cause du changement de signe)
       
       Serait-il utile d'ajouter une test b==1 pour supprimer une colonne?
       */
#define GCD_MAX_A 15
#define GCD_MAX_B 15
    /* la commutativite du pgcd n'est pas utilisee pour reduire la
       taille de la table */
    static Value 
	gcd_look_up[GCD_MAX_A+1][GCD_MAX_B+1]= {
	/*  b ==        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 */
        {/* a ==   0 */ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
        {/* a ==   1 */ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {/* a ==   2 */ 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1}, 
        {/* a ==   3 */ 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3},
        {/* a ==   4 */ 4, 1, 2, 1, 4, 1, 2, 1, 4, 1, 2, 1, 4, 1, 2, 1},
        {/* a ==   5 */ 5, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5},
        {/* a ==   6 */ 6, 1, 2, 3, 2, 1, 6, 1, 2, 3, 2, 1, 6, 1, 2, 3},
        {/* a ==   7 */ 7, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 7, 1},
        {/* a ==   8 */ 8, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1},
        {/* a ==   9 */ 9, 1, 1, 3, 1, 1, 3, 1, 1, 9, 1, 1, 3, 1, 1, 3},
        {/* a ==  10 */10, 1, 2, 1, 2, 5, 2, 1, 2, 1,10, 1, 2, 1, 2, 5},
        {/* a ==  11 */11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,11, 1, 1, 1, 1},
        {/* a ==  12 */12, 1, 2, 3, 4, 1, 6, 1, 4, 3, 2, 1,12, 1, 2, 3},
        {/* a ==  13 */13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,13, 1, 1},
        {/* a ==  14 */14, 1, 2, 1, 2, 1, 2, 7, 2, 1, 2, 1, 2, 1,14, 1},
        {/* a ==  15 */15, 1, 1, 3, 1, 5, 3, 1, 1, 3, 5, 1, 3, 1, 1, 15}
    };
    /* on pourrait aussi utiliser une table des nombres premiers
       pour diminuer le nombre de boucles */

    /* on utilise la valeur particuliere -1 pour iterer */
    Value gcd = VALUE_MONE;
    Value mod;

    assert(value_gt(a,b) && VALUE_POS_P(b));

    do{
	if(value_le(b,(Value)GCD_MAX_B) && 
	   value_le(a,(Value)GCD_MAX_A)) {
	    gcd = gcd_look_up[VALUE_TO_INT(a)][VALUE_TO_INT(b)];
	    break;
	}
	else {
	    /* compute modulo(a,b) en utilisant la routine C puisque a et b
	       sont strictement positifs (vaudrait-il mieux utiliser la
	       soustraction?) */
	    mod = value_mod(a,b);
	    if(VALUE_ZERO_P(mod)) {
		gcd = b;
	    }
	    else {
		a = b;
		b = mod;
	    }
	}
    } while(VALUE_NEG_P(gcd));

    return gcd;
}

/* int gcd_subtract(int a, int b): find the gcd (pgcd) of two integers
 *
 *	There is no precondition on the input. Negative input is handled
 *	in the same way as positive ones. If one input is zero the output
 *	is equal to the other input - thus an input of two zeros is the
 *	only way an output of zero is created.
 *
 *	Postcondition:	gcd(a,b) > 0 ; gcd(a,b)==0 iff a==0 and b==0
 *                      whereas it should be undefined (F. Irigoin)
 *
 *      Exception: gcd(0,0) aborts
 *
 *      Implementation: by subtractions
 *
 * Note: the signs of a and b do not matter because they can be exactly
 * divided by the gcd
 */
Value gcd_subtract(Value a, Value b)
{
    a = value_abs(a);
    b = value_abs(b);

    while (VALUE_NOTZERO_P(a) && VALUE_NOTZERO_P(b)) {
	if (value_ge(a,b)) 
	    a = value_minus(a,b);
	else
	    b = value_minus(b,a);
    }

    if (a == VALUE_ZERO) {
	assert(b!=VALUE_ZERO);
	return b;
    }
    else {
	/* b == 0 */
	assert(a!=VALUE_ZERO);
	return a; 
    }
}

/* int vecteur_bezout(int u[], int v[], int l): calcul du vecteur v
 * qui verifie le theoreme de bezout pour le vecteur u; les vecteurs u et
 * v sont de dimension l
 *
 *   ->  ->        ->
 * < u . v > = gcd(u )
 *              i
 */
Value vecteur_bezout(Value u[], Value v[], int l)
{
    Value gcd, a1, x;
    Value *p1, *p2;
    int i, j;

    assert(l>0);

    if (l==1) {
	v[0] = VALUE_ONE;
	gcd = u[0];
    }
    else {
	p1 = &v[0]; p2 = &v[1];
	a1 = u[0]; gcd = u[1];
	gcd = bezout(a1,gcd,p1,p2);

	/* printf("gcd = %d \n",gcd); */

	for (i=2;i<l;i++){ 
	    /* sum u * v = gcd(u ) 
             * k<l  k   k  k<l  k
	     *
	     * a1 = gcd  (u )
	     *      k<l-1  k
             */
	    a1 = u[i];
	    p1 = &v[i];
	    gcd = bezout(a1,gcd,p1,&x);
	    /* printf("gcd = %d\n",gcd); */
	    for (j=0;j<i;j++)
		v[j] = value_mult(v[j],x);	
	} 
    }

    return gcd;
}

/* int bezout(int a, int b, int *x, int *y): calcule x et y, les deux
 * nombres qui verifient le theoreme de Bezout pour a et b; le pgcd de
 * a et b est retourne par valeur
 *
 * a * x + b * y = gcd(a,b)
 * return gcd(a,b)
 */
Value bezout(Value a, Value b, Value *x, Value *y)
{
    Value u0=VALUE_ONE,u1=VALUE_ZERO,v0=VALUE_ZERO,v1=VALUE_ONE;
    Value a0,a1,u,v,r,q,c;

    if (value_ge(a,b))
    {
	a0 = a;
	a1 = b;
	c = VALUE_ZERO;
    }
    else
    {
	a0 = b;
	a1 = a;
	c = VALUE_ONE;
    }
	
    r = value_mod(a0,a1);
    while (r!=0)
    {
	q = value_div(a0,a1);
	u = u0 - value_mult(u1,q);

	v = v0 - value_mult(v1,q);
	a0 = a1; a1 = r;
	u0 = u1;u1 = u;
	v0 = v1; v1 = v;

	r = value_mod(a0,a1);
    }
  
    if (c==0) {
	*x = u1;
	*y = v1;
    }
    else {
	*x = v1;
	*y = u1;
    }

       return(a1);
}

/* int bezout_grl(int a, int b, int *x, int *y): calcule x et y, les deux
 * entiers quelconcs qui verifient le theoreme de Bezout pour a et b; le pgcd
 * de a et b est retourne par valeur
 *
 * a * x + b * y = gcd(a,b)
 * return gcd(a,b)
 * gcd () >=0
 * le pre et le post conditions de pgcd  sont comme la fonction gcd_subtract().
 * les situations speciaux sont donnes ci_dessous:
 *  si (a==0 et b==0)  x=y=0; gcd()=0,
 *  si (a==0)(ou b==0) x=1(ou -1) y=0 (ou x=0 y=1(ou -1)) 
 *  et gcd()=a(ou -a) (ou gcd()=b(ou -b))
 */
Value bezout_grl(Value a, Value b, Value *x, Value *y)
{
    Value u0=VALUE_ONE,u1=VALUE_ZERO,v0=VALUE_ZERO,v1=VALUE_ONE;
    Value a0,a1,u,v,r,q,c;
    Value sa,sb;               /* les signes de a et b */

    sa = sb = 1;
    if (VALUE_NEG_P(a)){
	sa = VALUE_MONE;
	a = value_uminus(a);
    }
    if (VALUE_NEG_P(b)){
	sb  = VALUE_MONE;
	b = value_uminus(b);
    }
    if (VALUE_ZERO_P(a) && VALUE_ZERO_P(b)){
	*x = VALUE_ONE;
	*y = VALUE_ONE;
	return VALUE_ZERO;
    }
    else if(VALUE_ZERO_P(a) || VALUE_ZERO_P(b)){
	if (VALUE_ZERO_P(a)){
	    *x = VALUE_ZERO;
	    *y = sb;
	    return(b);
	}
	else{
	    *x = sa;
	    *y = VALUE_ZERO;
	    return(a);
	}
    }
    else{

	if (a>=b)
	{
	    a0 = a;
	    a1 = b;
	    c = VALUE_ZERO;
	}
	else
	{
	    a0 = b;
	    a1 = a;
	    c = VALUE_ONE;
	}
	
	r = value_mod(a0,a1);
	while (VALUE_NOTZERO_P(r))
	{
	    q = value_div(a0,a1);
	    u = u0 - value_mult(u1,q);

	    v = v0 - value_mult(v1,q);
	    a0 = a1; a1 = r;
	    u0 = u1; u1 = u;
	    v0 = v1; v1 = v;

	    r = value_mod(a0,a1);
	}
  
	if (VALUE_ZERO_P(c)) {
	    *x = value_mult(sa,u1);
	    *y = value_mult(sb,v1);
	}
	else {
	    *x = value_mult(sa,v1);
	    *y = value_mult(sb,u1);
	}

	return a1;
    }
}

/* end of $RCSfile: pgcd.c,v $
 */
