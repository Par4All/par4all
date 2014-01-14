/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#ifndef lint
static char vcid_pip_traiter[] = "$Id$";
#endif /* lint */

#include <stdio.h>
#include "pip__type.h"
#include "pip__sol.h"
#include "pip__tab.h"

Entier sol_pgcd();

int llog(x)
Entier x;
{int n = 0;
 while(x) x >>= 1, n++;
 return(n);
}

int chercher(p, masque, n)
Tableau *p;
int masque, n;
{int i;
 for(i = 0; i<n; i++)
     if(p->row[i].flags & masque) break;
 return(i);
}

/* il est convenu que traiter ne doit modifier ni le tableau, ni le contexte;
   le tableau peut grandir en cas de coupure (+1 en hauteur et +1 en largeur
   si nparm != 0) et en cas de partage (+1 en hauteur)(seulement si nparm != 0).
   le contexte peut grandir en cas de coupure (+2 en hauteur et +1 en largeur)
   (seulement si nparm !=0) et en cas de partage (+1 en hauteur)(nparm !=0).
   On estime le nombre de coupures a llog(D) et le nombre de partages a
   ni.
*/

Tableau *expanser(tp, virt, reel, ncol, off, dh, dw)
Tableau *tp;
int virt, reel, ncol, off, dh, dw;
{
 int i, j, ff;
 char *q; Entier *pq;
 Tableau *rp;
 rp = tab_alloc(reel+dh, ncol+dw, virt);
 q = (char *)rp + 2 * sizeof(int) + (virt + reel + dh) * sizeof(struct L);
 pq = (Entier *) q;
 for(i = off; i<virt + reel; i++)
     {ff = Flag(rp, i) = Flag(tp, i-off);
      if(ff & Unit) rp->row[i].objet.unit = tp->row[i-off].objet.unit;
      else{rp->row[i].objet.val = pq;
           pq +=(ncol + dw);
	   for(j = 0; j<ncol; j++) Index(rp,i,j) = Index(tp,i-off,j);
	   for(j = ncol; j<ncol+dw; j++)Index(rp,i,j) = 0;
	  }
     }
 return(rp);
}

int exam_coef(tp, nvar, ncol, bigparm)
Tableau *tp;
int nvar, ncol, bigparm;
{int i, j;
 int ff, fff;
 Entier x;
 for(i = 0; i<tp->height; i++)
     {ff = Flag(tp,i);
      if(ff == 0) break;
      if(ff == Unknown)
	  {if(bigparm >= 0 && (x = Index(tp,i, bigparm)))
	      {if(x<0) {Flag(tp, i) = Minus;
			return(i);
		       }
	       else    Flag(tp, i) = Plus;
	       continue;
	      }
	   ff = Zero;
	   for(j = nvar; j<ncol; j++)
	       {x = Index(tp, i, j);
		if(x<0) fff = Minus;
		else if (x>0) fff = Plus;
		else fff = Zero;
		if(fff != Zero && fff != ff)
		    if(ff == Zero) ff = fff;
		    else {ff = Unknown;
			  break;
			 }
	       }
	   Flag(tp, i) = ff;
	   if(ff == Minus) return(i);
	  }
      }
 return(i);
}

void traiter();

void compa_test(tp, context, ni, nvar, nparm, nc)
Tableau *tp, *context;
int ni, nvar, nparm, nc;
{
 int i, j;
 int ff;
 int cPlus, cMinus, isCritic;
 Tableau *tPlus, *tMinus;
 Entier discr[MAXPARM];
 int p; char *q;
 if(nparm == 0) return;
 if(nparm >= MAXPARM) {
     fprintf(stderr, "Trop de parametres : %d\n", nparm);
     exit(1);
     }
 q = tab_hwm();
 for(i = 0; i<ni + nvar; i++)
     {ff = Flag(tp,i);
      if(ff & (Critic | Unknown))
	  {isCritic = True;
	   for(j = 0; j<nvar; j++) if(Index(tp, i, j) > 0)
		 {isCritic = False;
		  break;
		 }
	   for(j = 0; j < nparm; j++) discr[j] = Index(tp, i, j+nvar+1);
	   discr[nparm] = Index(tp, i, nvar)- (isCritic ? 0 : 1);
	   tPlus = expanser(context, nparm, nc, nparm+1, nparm, 1, 0);
	   Flag(tPlus, nparm+nc) = Unknown;
           for(j = 0; j<=nparm; j++)Index(tPlus, nparm+nc, j) = discr[j];
	   p = sol_hwm();
	   traiter(tPlus, NULL, True, UN, nparm, 0, nc+1, 0, -1);
	   cPlus = is_not_Nil(p);
	   sol_reset(p);
	   for(j = 0; j<nparm+1; j++) discr[j] = -discr[j];
	   discr[nparm] = discr[nparm] - (isCritic ? 1 : 2);
	   tMinus = expanser(context, nparm, nc, nparm+1, nparm, 1, 0);
	   Flag(tMinus, nparm+nc) = Unknown;
	   for(j = 0; j<= nparm; j++)Index(tMinus, nparm+nc, j) = discr[j];
	   traiter(tMinus, NULL, True, UN, nparm, 0, nc+1, 0, -1);
	   cMinus = is_not_Nil(p);
	   sol_reset(p);
	   if (cPlus && cMinus)
	       Flag(tp,i) = isCritic ? Critic : Unknown;
	   else if (cMinus)
	      {Flag(tp,i) = Minus;
	       break;
	      }
	   else Flag(tp,i) = cPlus ? Plus : Zero;
	  }
     }
 tab_reset(q);
 return;
}

Entier valeur(tp, i, j, D)
Tableau *tp;
int i, j;
Entier D;
{
 if(Flag(tp, i) & Unit)
     return(tp->row[i].objet.unit == j ? D : 0);
 else return(Index(tp, i, j));
}

void solution(tp, nvar, nparm, D)
Tableau *tp;
int nvar, nparm;
Entier D;
{int i, j;
 int ncol = nvar + nparm + 1;
 sol_list(nvar);
 for(i = 0; i<nvar; i++)
     {sol_forme(nparm+1);
      for(j = nvar+1; j<ncol; j++)
	  sol_val(valeur(tp, i, j, D), D);
      sol_val(valeur(tp, i, nvar, D), D);
     }
}

int choisir_piv(tp, pivi, nvar, nligne, D)
Tableau *tp;
int pivi, nvar, nligne;
Entier D;
{
 int j, k;
 long pivot, foo, x;
 int pivj = -1;
 for(j = 0; j<nvar; j++)
     {if((foo = Index(tp, pivi, j)) <= 0) continue;
      if(pivj < 0)
	  {pivj = j;
	   pivot = foo;
	   continue;
	  }
      for(k = 0; k<nligne; k++)
          {x = pivot * valeur(tp, k, j, D) - valeur(tp, k, pivj, D) * foo;
           cross_product++;
	   if(x) break;
	  }
      if(x < 0)
	  {pivj = j;
	   pivot = foo;
	  }
     }
/* fprintf(stderr, "%d ", pivj); */
 return(pivj);
}

int pivoter(tp, pivi, D, nvar, nparm, ni)
Tableau *tp;
int pivi;
Entier D;
int nvar, nparm, ni;
{int pivj;
 int ncol = nvar + nparm + 1;
 int nligne = nvar + ni;
 int j, k;
 int x;
 int ff, fff;
 long int pivot, foo, z;
 Entier new[MAXCOL], *p;
 if(ncol >= MAXCOL) {
     fprintf(stderr, "Trop de colonnes : %d\n", ncol);
     exit(1);
     }
 if(verbose >0)
   tab_display(tp, dump);
 pivj = choisir_piv(tp, pivi, nvar, nligne, D);
 if(pivj < 0) return(-1);
 pivot = Index(tp, pivi, pivj);
 for(j = 0; j<ncol; j++) new[j] = (j == pivj ? D : -Index(tp, pivi, j));
 for(k = 0; k<nligne; k++)
     {if(Flag(tp,k) & Unit)continue;
      if(k == pivi)continue;
      foo = Index(tp, k, pivj);
      for(j = 0; j<ncol; j++)
	   {if(j == pivj) continue;
	    cross_product++;
	    z = Index(tp, k, j) * pivot - Index(tp, pivi, j) * foo;
	    if(z%D || z > 32767L * D)
	        {fprintf(stderr, "%ld/Catastrophe en li %d co %d "
			          ,cross_product, k, j);
		 fprintf(stderr, "%ld", z);
		 fprintf(stderr, FORMAT, D);
		 fprintf(stderr, "\n");
	         exit(30);
	        }
	    Index(tp, k, j) = z/D;
	   }
      }
 p = tp->row[pivi].objet.val;
 for(k = 0; k<nligne; k++)
     if((Flag(tp, k) & Unit) && tp->row[k].objet.unit == pivj) break;
 Flag(tp, k) = Plus;
 tp->row[k].objet.val = p;
 for(j = 0; j<ncol; j++) *p++ = new[j];
 Flag(tp, pivi) = Unit | Zero;
 tp->row[pivi].objet.unit = pivj;

 for(k = 0; k<nligne; k++)
    {ff = Flag(tp, k);
     if(ff & Unit) continue;
     x = Index(tp, k, pivj);
     if(x < 0) fff = Minus;
     else if(x == 0) fff = Zero;
     else fff = Plus;
     if(fff != Zero && fff != ff)
         if(ff == Zero) ff = (fff == Minus ? Unknown : fff);
	 else ff = Unknown;
     Flag(tp, k) = ff;
    }
/* if((cross_product % 10000) == 0)
     fprintf(stderr,"%ld pivotements\n\r", cross_product); */
 return((Entier) pivot);
}

/* dans cette version, "traiter" modifie ineq; par contre
   le contexte est immediatement recopie' */

void traiter(tp, ctxt, iq, D, nvar, nparm, ni, nc, bigparm)
Tableau *tp, *ctxt;
int iq, nvar, nparm, ni, nc, bigparm;
Entier D;
{
 int j;
 int pivi, nligne, ncol;
 char *x;
 Tableau *context;
 int dch, dcw;
 dcw = llog(D);
 dch = 2 * dcw + 1;
 x = tab_hwm();
 context = expanser(ctxt, 0, nc, nparm+1, 0, dch, dcw);
 for(;;)
     {nligne = nvar+ni; ncol = nvar+nparm+1;
      if(nligne > tp->height || ncol > tp->width)
	  {fprintf(stderr, "%ld/Explosion ! :trt\n", cross_product);
	   exit(69);
	  }
      pivi = chercher(tp, Minus, nligne);
      if(pivi < nligne) goto pirouette;		/* il y a une ligne connue
      						   negative */
      pivi = exam_coef(tp, nvar, ncol, bigparm);
      if(pivi < nligne) goto pirouette;		/* il y a une ligne ou tous
      						   les coefficients sont
						   negatif */
      compa_test(tp, context, ni, nvar, nparm, nc);
      pivi = chercher(tp, Minus, nligne);
      if(pivi < nligne) goto pirouette;		/* on trouve une ligne negative
      						  apres test de compatibilite'*/
      pivi = chercher(tp, Critic, nligne);
      if(pivi >= nligne)pivi = chercher(tp, Unknown, nligne);
      if(pivi < nligne)
          {		/* on a trouve' une ligne de signe indetermine',
	  		   et donc on divise le probleme en deux */
	   Tableau * ntp;
	   Entier discr[MAXPARM], com_dem;
           char *q;
	   if(nc >= context->height)
	       {dcw = llog(D);
		dch = 2 * dcw + 1;
		context = expanser(context, 0, nc, nparm+1, 0, dch, dcw);
	       }
	   if(nparm >= MAXPARM) {
	        fprintf(stderr, "Trop de parametres : %d\n", nparm);
		exit(2);
		}
           q = tab_hwm();
	   ntp = expanser(tp, nvar, ni, ncol, 0, 0, 0);
	   sol_if();
	   sol_forme(nparm+1);
	   com_dem = 0;
	   for(j = 0; j<nparm; j++)
	       {discr[j] = Index(tp, pivi, j + nvar +1);
	        com_dem = sol_pgcd(com_dem, discr[j]);
               }
           discr[nparm] = Index(tp, pivi, nvar);
	   com_dem = sol_pgcd(com_dem, discr[nparm]);
           if(com_dem < 0)
	       {printf("pgcd negatif ! %d : trt\n", com_dem);
	        exit(88);
	       }
           for(j = 0; j<=nparm; j++)
	       {discr[j] /= com_dem;
	        Index(context, nc, j) = discr[j];
		sol_val(discr[j], UN);
	       }
	   Flag(context, nc) = Unknown;
	   Flag(ntp, pivi) = Plus;
	   traiter(ntp, context, iq, D, nvar, nparm, ni, nc+1, bigparm);
	   tab_reset(q);
	   for(j = 0; j<nparm; j++) discr[j] = -discr[j];
	   discr[nparm] = -discr[nparm] - 1;
           Flag(tp, pivi) = Minus;
	   for(j = 0; j<=nparm; j++)
               Index(context, nc, j) = discr[j];
	   nc++;
	   goto pirouette;
	  }
/* Ici, on a trouve' une solution. Est-elle entiere ? */	  
	  
/*    if(non_borne(tp, nvar, D, bigparm))
	    {
             sol_nil();
	     break;
	    }                                           */

            if(iq){pivi = integrer(&tp, &context, D,
			     &nvar, &nparm, &ni, &nc);
             if(pivi >= 0) goto pirouette;
	    }

/* Oui, elle est entiere, ou bien on s'en fiche (iq == 0) */
	    
      solution(tp, nvar, nparm, D);
      break;

/* dans tout les cas ou on a trouve' une ligne ne'gative, on vient ici
   pour effectuer l'echange variable dependante <-> variable independante */

pirouette :
      if((D = pivoter(tp, pivi, D, nvar, nparm, ni)) < 0)
          {
	   sol_nil();
	   break;
	  }
     }
 tab_reset(x);
}
