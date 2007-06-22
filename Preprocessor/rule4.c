/* Check that it exercises Rule 4 for function declaration... but the
   problem in SPEC2000/ammp.c is not reproduced. */

typedef struct{
	double  x,y,z;
	double  fx,fy,fz;
	int serial;
	double  q,a,b,mass;
	void *next;
	char active;
	char name[9];
	double  chi,jaa;
	double  vx,vy,vz,vw,dx,dy,dz;
	double  gx,gy,gz;
	double  VP,px,py,pz,dpx,dpy,dpz; 
	double  qxx,qxy,qxz,qyy,qyz,qzz;
	void *close[200 ];
	void *excluded[32 ];
	char exkind[32 ];
	int  dontuse;
	} ATOM;

typedef struct{
	ATOM *atom1,*atom2,*atom3;
	double  target,k;
	void *next;
	}  ANGLE;

int f_ho_angle(lambda)
double  lambda;
/*  returns 0 if error, 1 if OK */
{
	ANGLE *bp;
	double  r,k,ux1,uy1,uz1,ux2,uy2,uz2;
	ATOM *a1,*a2,*a3;
	double  x1,y1,z1,x2,y2,z2;
	double  r1,r2,dtheta,dp;
	double  r11,r22,sdth;
	double  hol,get_f_variable(),target;

	hol = get_f_variable("lambda");
	if( hol < 0. ) hol = 0.;
	if( hol > 1. ) hol = 1.;
	bp = angle_first;
       if( bp == 0  ) return 1;
       while(1)
       {
	if( bp == 0 ) return 0;
	k = bp->k;
	a1 = bp->atom1; a2 = bp->atom2; a3 = bp->atom3;
	if( a1->active|| a2->active || a3->active )
	{
	x1 = (a1->x -a2->x +lambda*(a1->dx-a2->dx));
	y1 = (a1->y -a2->y +lambda*(a1->dy-a2->dy));
	z1 = (a1->z -a2->z +lambda*(a1->dz-a2->dz));
	x2 = (a3->x -a2->x +lambda*(a3->dx-a2->dx));
	y2 = (a3->y -a2->y +lambda*(a3->dy-a2->dy));
	z2 = (a3->z -a2->z +lambda*(a3->dz-a2->dz));
	dp = x1*x2+y1*y2+z1*z2;
	r1 = sqrt(x1*x1+y1*y1+z1*z1);
	r2 = sqrt(x2*x2+y2*y2+z2*z2);
	if( r1 < 1.e-5 || r2 < 1.e-5) goto SKIP;
	r = r1*r2;
	if( r > 1.e-8){

	dp = dp/r;  if( dp > 1.) dp = 1.; if( dp < -1.) dp = -1.;
	dtheta = acos(dp);
	target = hol*dtheta + (1.-hol)*bp->target;
	sdth = sin(dtheta); if( sdth < 1.e-3) sdth = 1.e-3;
	r11 = r1*sdth; r22 = r2*sdth;
	ux1 = x2/r2 - dp*x1/r1;
	uy1 = y2/r2 - dp*y1/r1;
	uz1 = z2/r2 - dp*z1/r1;
	ux2 = x1/r1 - dp*x2/r2;
	uy2 = y1/r1 - dp*y2/r2;
	uz2 = z1/r1 - dp*z2/r2;
	dtheta = -2.*k*(target - dtheta)*(1.-hol);
	ux1 = ux1*dtheta/r11;
	uy1 = uy1*dtheta/r11;
	uz1 = uz1*dtheta/r11;
	ux2 = ux2*dtheta/r22;
	uy2 = uy2*dtheta/r22;
	uz2 = uz2*dtheta/r22;
	if( a1->active)
	{
	a1->fx += ux1;
	a1->fy += uy1;
	a1->fz += uz1;
	}

	if( a2->active)
	{
	a2->fx += -ux1 - ux2;
	a2->fy += -uy1 - uy2;
	a2->fz += -uz1 - uz2;
	}

	if( a3->active)
	{
	a3->fx += ux2;
	a3->fy += uy2;
	a3->fz += uz2;
	}

	}	
	}	
SKIP:
	if( bp == bp->next ) return 1;
	bp = bp->next;
       }
}
