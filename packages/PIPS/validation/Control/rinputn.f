C     Initially from renault.f. The bug was SQRT(SQRT(X)) in an expression.
C     Modified to get rid of subroutines and functions

      subroutine rinput
c  
c
c - MODIFICATION CONCERNANT LA LECTURE DE LA VARIABLE ENTIERE MSHEXT
c   DANS LE FICHIER 5 APRES LA LECTURE DE NZ.
c - APRES LA LECTURE DE CYL, SI MSHEXT=1, ON FORCE CYL EGAL A ZERO
c - ON CONTOURNE LE CALCUL DE CERTAINES QUANTITES GEO. (SKIPCO ETC...)
c
c ======================================================================
c
c      rinput reads the input data file and computes derived quantities,
c      scalars, plot data, and the species and reaction tabular data.
c
c      rinput is called by:  kiva
c
c      rinput calls the following subroutines and functions:  fuel and
c             tran3d, and the system routines sample and sampon, which
c             are described in the epilog.
c
c ======================================================================
c
c      INCLUDE '../include/comd.h'
C
C ======================================================================
C
C           VERSION MODIFIEE ET CORRIGEE DE KIVA2 (KIVA2.UPD)
C
C                VERSION RECUE PAR RENAULT LE 5 NOV 90
C
C ======================================================================
C
C SIGNIFICATION DES VARIABLES :
C ---------------------------
C
C  NV     = MAXIMUM NO. OF VERTICES, =(NX+1)*(NY+1)*(NZ+1)
C  LNXPYP = MAXIMUM NO. OF VERTICES IN A PLANE, =(NX+1)*(NY+1)
C  LNSP   = MAXIMUM NO. OF SPECIES, MUST BE .GE. 3 WHEN CHOPPER IS USED
C  LNRK   = MAXIMUM NO. OF KINETIC CHEMICAL REACTIONS
C  LNRE   = MAXIMUM NO. OF EQUILIBRIUM CHEMICAL REACTIONS
C  NPAR   = MAXIMUM NO. OF PARTICLES PRESENT AT ANY ONE TIME
C  LP     = MAXIMUM NO. OF POINTS USED TO DEFINE PISTON SILHOUETTE
C           OR HEAD SILHOUETTE
C  LV     = MAX. NO. OF VIEWS OF EACH 3-D PLOT TYPE (ZONE,VECTOR,CONTOUR)
C  LVAP   = MAX. NO. OF ENTRIES IN LIQUID VAPOR PRES. TABLES IN FUELIB
C  LNZP   = MAXIMUM NO. OF PLANES, =(NZ+1)
C  LVEL   = MAX. NO. OF ENTRIES IN THE INJECTION VELOCITY TABLE
C  LNOZ   = MAX. NO. OF FUEL INJECTION NOZZLES (=1 IF NY=1)
C  LNPT1  = MAX. NO. DE POINTS DEFINISSANT LA LOI DE LEVEE: ZXL=F(ANGLE)
C  LNPT2  = MAX. NO. DE POINTS A L'ENTREE DE LA PIPE D'ADMISSION
C
C ===============PARAMETRE POUR OPTIMISATION DE LA CLASSE ==============
C
C MOYENC
C
      PARAMETER (NV=24000,LNXPYP=1700,LNSP=12,LNRK=4,LNRE=6,
C
C MOYEND
C
C      PARAMETER (NV=127500,LNXPYP=2500,LNSP=12,LNRK=4,LNRE=6,
C
C MOYENE
C
C      PARAMETER (NV=240000,LNXPYP=3500,LNSP=12,LNRK=4,LNRE=6,
C
C NUIT
C
C      PARAMETER (NV=340000,LNXPYP=6500,LNSP=12,LNRK=4,LNRE=6,
C
C ======================================================================
C

     &           NPAR=2000,LP=35,LV=5,LVAP=74,LNZP=NV/LNXPYP,LPLUS=60)

      PARAMETER (LVEL=100,LNOZ=12)
      PARAMETER (LNPT1=200,LNPT2=200,LNPDB=600)
      PARAMETER (NVXPYP = NV + LNXPYP + LPLUS )

c      INCLUDE 'common.h'
      logical   cropen,vopen
c
c +++ note: on short word-length machines, real numbers must be declared
c +++       double precision (60-64 bits required); however, 32 bits
c +++       will suffice for integers; this is why they are separated
c +++       from the real numbers in the common blocks below.
c
c +++ note: the common blocks are written with the implicit assumption
c +++       that they will be loaded exactly as they appear below.
c +++       because aa- and zz- names often span more than one common,
c +++       problems will result if the loader scatters the commons.
c +++       in this case, the loader must be instructed to load the
c +++       commons as a single module.  if this is not an option,
c +++       the user must restructure the commons and make appropriate
c +++       changes in subroutines begin, taperd, and tapewr.
c  
      common /lc1/ aaa1(1),x(-lnxpyp:nvxpyp),y(-lnxpyp:nvxpyp),
     1 z(-lnxpyp:nvxpyp),u(-lnxpyp:nvxpyp),v(-lnxpyp:nvxpyp),
     2 w(-lnxpyp:nvxpyp),ro(-lnxpyp:nvxpyp),vol(-lnxpyp:nvxpyp),
     3 p(-lnxpyp:nvxpyp),amu(-lnxpyp:nvxpyp),f(-lnxpyp:nvxpyp),
     4 fv(-lnxpyp:nvxpyp),temp(-lnxpyp:nvxpyp),sie(-lnxpyp:nvxpyp),
     5 zzz1
C
      common /lc1i/ iaaa1(1),
     & itab(-lnxpyp:nvxpyp),jtab(-lnxpyp:nvxpyp),ktab(-lnxpyp:nvxpyp),
     1 nzzz1
C
CCC      common /lc1z/ zzz1
C
      common /lc2/ aaa2(1),tke(-lnxpyp:nvxpyp),pit(-lnxpyp:nvxpyp),
     1 pit1(-lnxpyp:nvxpyp),un(-lnxpyp:nvxpyp),vn(-lnxpyp:nvxpyp),
     2 wn(-lnxpyp:nvxpyp),suvw(-lnxpyp:nvxpyp),zzz2
C
      common /lc3/ aaa3(1),e1(-lnxpyp:nvxpyp),e2(-lnxpyp:nvxpyp),
     1 e3(-lnxpyp:nvxpyp),e4(-lnxpyp:nvxpyp),e5(-lnxpyp:nvxpyp),
     2 e6(-lnxpyp:nvxpyp),e7(-lnxpyp:nvxpyp),e8(-lnxpyp:nvxpyp),
     3 e9(-lnxpyp:nvxpyp),e10(-lnxpyp:nvxpyp),e11(-lnxpyp:nvxpyp),
     4 e12(-lnxpyp:nvxpyp),e13(-lnxpyp:nvxpyp),e14(-lnxpyp:nvxpyp),
     5 e15(-lnxpyp:nvxpyp),e16(-lnxpyp:nvxpyp),e17(-lnxpyp:nvxpyp),
     6 e18(-lnxpyp:nvxpyp),e19(-lnxpyp:nvxpyp),e20(-lnxpyp:nvxpyp),zzz3
C
      common /lc4/ aaa4(1),e21(-lnxpyp:nvxpyp),e22(-lnxpyp:nvxpyp),
     1 e23(-lnxpyp:nvxpyp),e24(-lnxpyp:nvxpyp),e25(-lnxpyp:nvxpyp),
     2 e26(-lnxpyp:nvxpyp),e27(-lnxpyp:nvxpyp),e28(-lnxpyp:nvxpyp),
     3 e29(-lnxpyp:nvxpyp),e30(-lnxpyp:nvxpyp),e31(-lnxpyp:nvxpyp),
     4 e32(-lnxpyp:nvxpyp),e33(-lnxpyp:nvxpyp),e34(-lnxpyp:nvxpyp),
     5 e35(-lnxpyp:nvxpyp),e36(-lnxpyp:nvxpyp),e37(-lnxpyp:nvxpyp),
     6 e38(-lnxpyp:nvxpyp),e39(-lnxpyp:nvxpyp),e40(-lnxpyp:nvxpyp),zzz4
C
      common /lc5/ aaa5(1),e41(-lnxpyp:nvxpyp),e42(-lnxpyp:nvxpyp),
     1 e43(-lnxpyp:nvxpyp),e44(-lnxpyp:nvxpyp),e45(-lnxpyp:nvxpyp),
     2 e46(-lnxpyp:nvxpyp),e47(-lnxpyp:nvxpyp),e48(-lnxpyp:nvxpyp),
     3 e49(-lnxpyp:nvxpyp),e50(-lnxpyp:nvxpyp),e51(-lnxpyp:nvxpyp),
     4 e52(-lnxpyp:nvxpyp),e53(-lnxpyp:nvxpyp),e54(-lnxpyp:nvxpyp),
     5 e55(-lnxpyp:nvxpyp),e56(-lnxpyp:nvxpyp),e57(-lnxpyp:nvxpyp),
     6 e58(-lnxpyp:nvxpyp),e59(-lnxpyp:nvxpyp),e60(-lnxpyp:nvxpyp),
     7 e61(-lnxpyp:nvxpyp),e62(-lnxpyp:nvxpyp),e63(-lnxpyp:nvxpyp),
     8 e64(-lnxpyp:nvxpyp),zzz5
C
      common /lc6r/ aaa6(1),spd(-lnxpyp:nvxpyp,lnsp),htform(lnsp),
     1 mw(lnsp),rhoi(lnsp),rmw(lnsp),cb(lnrk),cf(lnrk),eb(lnrk),
     2 ef(lnrk),nelem(lnrk),qr(lnrk),zetab(lnrk),zetaf(lnrk),
     3 ae(lnsp,lnrk),be(lnsp,lnrk),as(lnre),bs(lnre),cs(lnre),ds(lnre),
     4 es(lnre),qeq(lnre),tspm(lnsp),fam(lnsp,lnrk),fbm(lnsp,lnrk),
     5 fbmam(lnsp,lnrk),fbnan(lnsp,lnre),wdd(lnzp,lnsp),
     8 spdin0(lnsp),roin(lnxpyp),spdin(lnxpyp,lnsp),
     9 spdamb(lnsp),rosiein(lnxpyp),zdmax(lnxpyp),zzz6
c
      common /lc6i/ aaa6i(1),
     1 am(lnsp,lnrk),bm(lnsp,lnrk),cm(lnsp,lnrk),
     2 an(lnsp,lnre),bn(lnsp,lnre),cn(lnsp,lnre),nlm(lnre),zzz6i
c
c      common /lc6z/ zzz6
C
C
      equivalence (e1,mv,rmv),(e2,gamma,rgamma),(e3,eps,scl),
     1 (e4,alx),(e5,aly),(e6,alz),(e7,afx),(e8,afy),(e9,afz),
     2 (e10,abx),(e11,aby),(e12,abz),(e13,cli),(e14,clj),(e15,clk),
     3 (e16,cfi),(e17,cfj),(e18,cfk),(e19,cbi),(e20,cbj),(e21,cbk)
      equivalence (e22,fsum14,rfsum14,xo),
     1 (e23,fsum34,rfsum34,yo),
     2 (e24,fsum84,rfsum84,zo),
     3 (e25,sum3,yspm,dudx,ual),
     4 (e26,sum2,yspd,dudy,cv,uaf),
     5 (e27,conq,dudz,r,uab),
     6 (e28,vapm,xcen,dvdx,sietil,rpa),
     7 (e29,enth0,ycen,dvdy,ptem),
     8 (e30,zcen,dvdz,ml,rmldt),
     9 (e31,xl,dwdx,cvterm,mf,rmfdt),
     x (e32,yl,dwdy,mb,rmbdt),
     1 (e33,zl,dwdz),
     2 (e34,phid,fxl),
     3 (e35,enthdf,disptil,fxf),
     4 (e36,spd14,dyp14,tem14,resu,uala,fxb),
     5 (e37,augmv,spd34,dyp34,tem34,resv,uafa,voll,totie),
     6 (e38,ru,spd84,dyp84,tem84,resw,uaba,rosie,umom),
     7 (e39,rv,hisp,util,rotke,vmom),
     8 (e40,rw,spmtil,vtil,roscl,wmom)
      equivalence (e41,enthtil,wtil,rosiev,smom,ttke),
     1 (e42,ddy,rmvsu,rdrde,rotkev,fxv,teps),
     2 (e43,res,resuo,rosclv,fxvm,zchop),
     3 (e44,resold,resvo,mvp,znwchp),
     4 (e45,dres,reswo,mp),
     5 (e46,rdrdy,dresu,rdrdt,rdrdp,rdrdk,fxlm,work1),
     6 (e47,deltay,dresv,dtemp,dp,deltke,deps,fxfm,work2),
     7 (e48,tke14,dresw,fxbm,work3),
     8 (e49,tke34,pn,drds,duds),
     9 (e50,tke84,phip,dtds,dvds),
     x (e51,eps14,ub,dwds),
     1 (e52,eps34,vb,dsds),
     2 (e53,eps84,wb,dvol),
     3 (e54,totcm,dissip,fxi),
     4 (e55,dmtot,duhat,htc,rtermk,fxj),
     5 (e56,toth,dvhat,volb,rterme,fxk),
     6 (e57,dsiep,dwhat,rrovoll,s),
     7 (e58,dtkep,cpc),
     8 (e59,ron,htctil)
      equivalence (e60,sien,ttil),(e61,tketil),(e62,epstil),(e63,tken),
     1            (e64,epsn)
      equivalence (spd,spm)
c
      dimension mv(-lnxpyp:nvxpyp),rmv(-lnxpyp:nvxpyp),
     1 gamma(-lnxpyp:nvxpyp),rgamma(-lnxpyp:nvxpyp),
     2 eps(-lnxpyp:nvxpyp),scl(-lnxpyp:nvxpyp),
     3 alx(-lnxpyp:nvxpyp),aly(-lnxpyp:nvxpyp),alz(-lnxpyp:nvxpyp),
     4 afx(-lnxpyp:nvxpyp),afy(-lnxpyp:nvxpyp),afz(-lnxpyp:nvxpyp),
     5 abx(-lnxpyp:nvxpyp),aby(-lnxpyp:nvxpyp),abz(-lnxpyp:nvxpyp),
     6 cli(-lnxpyp:nvxpyp),clj(-lnxpyp:nvxpyp),clk(-lnxpyp:nvxpyp),
     7 cfi(-lnxpyp:nvxpyp),cfj(-lnxpyp:nvxpyp),cfk(-lnxpyp:nvxpyp),
     8 cbi(-lnxpyp:nvxpyp),cbj(-lnxpyp:nvxpyp),cbk(-lnxpyp:nvxpyp)
      dimension fsum14(-lnxpyp:nvxpyp),rfsum14(-lnxpyp:nvxpyp),
     1 xo(-lnxpyp:nvxpyp),fsum34(-lnxpyp:nvxpyp),
     2 rfsum34(-lnxpyp:nvxpyp),yo(-lnxpyp:nvxpyp),
     3 fsum84(-lnxpyp:nvxpyp),rfsum84(-lnxpyp:nvxpyp),
     4 zo(-lnxpyp:nvxpyp),sum3(-lnxpyp:nvxpyp),yspm(-lnxpyp:nvxpyp),
     5 dudx(-lnxpyp:nvxpyp),ual(-lnxpyp:nvxpyp),sum2(-lnxpyp:nvxpyp),
     6 yspd(-lnxpyp:nvxpyp),dudy(-lnxpyp:nvxpyp),cv(-lnxpyp:nvxpyp),
     7 uaf(-lnxpyp:nvxpyp),conq(-lnxpyp:nvxpyp),dudz(-lnxpyp:nvxpyp),
     8 r(-lnxpyp:nvxpyp),uab(-lnxpyp:nvxpyp),vapm(-lnxpyp:nvxpyp),
     9 xcen(-lnxpyp:nvxpyp),dvdx(-lnxpyp:nvxpyp),sietil(-lnxpyp:nvxpyp),
     x rpa(-lnxpyp:nvxpyp),enth0(-lnxpyp:nvxpyp),ycen(-lnxpyp:nvxpyp),
     1 dvdy(-lnxpyp:nvxpyp),ptem(-lnxpyp:nvxpyp),zcen(-lnxpyp:nvxpyp),
     2 dvdz(-lnxpyp:nvxpyp),ml(-lnxpyp:nvxpyp),rmldt(-lnxpyp:nvxpyp),
     3 xl(-lnxpyp:nvxpyp),dwdx(-lnxpyp:nvxpyp),cvterm(-lnxpyp:nvxpyp),
     4 mf(-lnxpyp:nvxpyp),rmfdt(-lnxpyp:nvxpyp),yl(-lnxpyp:nvxpyp),
     5 dwdy(-lnxpyp:nvxpyp),mb(-lnxpyp:nvxpyp),rmbdt(-lnxpyp:nvxpyp),
     6 zl(-lnxpyp:nvxpyp),dwdz(-lnxpyp:nvxpyp),phid(-lnxpyp:nvxpyp),
     7 fxl(-lnxpyp:nvxpyp),enthdf(-lnxpyp:nvxpyp),
     8 disptil(-lnxpyp:nvxpyp),fxf(-lnxpyp:nvxpyp),
     9 spd14(-lnxpyp:nvxpyp),dyp14(-lnxpyp:nvxpyp)
       dimension tem14(-lnxpyp:nvxpyp),resu(-lnxpyp:nvxpyp),
     1 uala(-lnxpyp:nvxpyp),fxb(-lnxpyp:nvxpyp),augmv(-lnxpyp:nvxpyp),
     2 spd34(-lnxpyp:nvxpyp),dyp34(-lnxpyp:nvxpyp),
     3 tem34(-lnxpyp:nvxpyp),resv(-lnxpyp:nvxpyp),uafa(-lnxpyp:nvxpyp),
     4 voll(-lnxpyp:nvxpyp),totie(-lnxpyp:nvxpyp),ru(-lnxpyp:nvxpyp),
     5 spd84(-lnxpyp:nvxpyp),dyp84(-lnxpyp:nvxpyp),
     6 tem84(-lnxpyp:nvxpyp),resw(-lnxpyp:nvxpyp),uaba(-lnxpyp:nvxpyp),
     7 rosie(-lnxpyp:nvxpyp),umom(-lnxpyp:nvxpyp)
      dimension rv(-lnxpyp:nvxpyp),hisp(-lnxpyp:nvxpyp),
     1 util(-lnxpyp:nvxpyp),rotke(-lnxpyp:nvxpyp),vmom(-lnxpyp:nvxpyp),
     2 rw(-lnxpyp:nvxpyp),spmtil(-lnxpyp:nvxpyp),vtil(-lnxpyp:nvxpyp),
     3 roscl(-lnxpyp:nvxpyp),wmom(-lnxpyp:nvxpyp)
      dimension enthtil(-lnxpyp:nvxpyp),wtil(-lnxpyp:nvxpyp),
     1 rosiev(-lnxpyp:nvxpyp),smom(-lnxpyp:nvxpyp),ttke(-lnxpyp:nvxpyp),
     2 ddy(-lnxpyp:nvxpyp),rmvsu(-lnxpyp:nvxpyp),rdrde(-lnxpyp:nvxpyp),
     3 rotkev(-lnxpyp:nvxpyp),fxv(-lnxpyp:nvxpyp),teps(-lnxpyp:nvxpyp),
     4 res(-lnxpyp:nvxpyp),resuo(-lnxpyp:nvxpyp),rosclv(-lnxpyp:nvxpyp),
     5 fxvm(-lnxpyp:nvxpyp),zchop(-lnxpyp:nvxpyp),
     6 resold(-lnxpyp:nvxpyp),resvo(-lnxpyp:nvxpyp),mvp(-lnxpyp:nvxpyp),
     7 znwchp(-lnxpyp:nvxpyp),dres(-lnxpyp:nvxpyp),
     8 reswo(-lnxpyp:nvxpyp),mp(-lnxpyp:nvxpyp),rdrdy(-lnxpyp:nvxpyp),
     9 dresu(-lnxpyp:nvxpyp),rdrdt(-lnxpyp:nvxpyp),
     x rdrdp(-lnxpyp:nvxpyp),rdrdk(-lnxpyp:nvxpyp),fxlm(-lnxpyp:nvxpyp),
     1 work1(-lnxpyp:nvxpyp),deltay(-lnxpyp:nvxpyp),
     2 dresv(-lnxpyp:nvxpyp),dtemp(-lnxpyp:nvxpyp),dp(-lnxpyp:nvxpyp),
     3 deltke(-lnxpyp:nvxpyp),deps(-lnxpyp:nvxpyp),fxfm(-lnxpyp:nvxpyp),
     4 work2(-lnxpyp:nvxpyp),tke14(-lnxpyp:nvxpyp),
     5 dresw(-lnxpyp:nvxpyp),fxbm(-lnxpyp:nvxpyp),work3(-lnxpyp:nvxpyp),
     6 tke34(-lnxpyp:nvxpyp),pn(-lnxpyp:nvxpyp),drds(-lnxpyp:nvxpyp),
     7 duds(-lnxpyp:nvxpyp),tke84(-lnxpyp:nvxpyp),phip(-lnxpyp:nvxpyp),
     8 dtds(-lnxpyp:nvxpyp),dvds(-lnxpyp:nvxpyp),eps14(-lnxpyp:nvxpyp),
     9 ub(-lnxpyp:nvxpyp),dwds(-lnxpyp:nvxpyp),eps34(-lnxpyp:nvxpyp)
       dimension vb(-lnxpyp:nvxpyp),dsds(-lnxpyp:nvxpyp),
     1 eps84(-lnxpyp:nvxpyp),wb(-lnxpyp:nvxpyp),dvol(-lnxpyp:nvxpyp),
     2 totcm(-lnxpyp:nvxpyp),dissip(-lnxpyp:nvxpyp),fxi(-lnxpyp:nvxpyp),
     3 dmtot(-lnxpyp:nvxpyp),duhat(-lnxpyp:nvxpyp),htc(-lnxpyp:nvxpyp),
     4 rtermk(-lnxpyp:nvxpyp),fxj(-lnxpyp:nvxpyp),toth(-lnxpyp:nvxpyp),
     5 dvhat(-lnxpyp:nvxpyp),volb(-lnxpyp:nvxpyp),
     6 rterme(-lnxpyp:nvxpyp),fxk(-lnxpyp:nvxpyp),dsiep(-lnxpyp:nvxpyp),
     7 dwhat(-lnxpyp:nvxpyp),rrovoll(-lnxpyp:nvxpyp),s(-lnxpyp:nvxpyp),
     8 dtkep(-lnxpyp:nvxpyp),cpc(-lnxpyp:nvxpyp),ron(-lnxpyp:nvxpyp),
     9 htctil(-lnxpyp:nvxpyp)
      dimension sien(-lnxpyp:nvxpyp),ttil(-lnxpyp:nvxpyp),
     1 tketil(-lnxpyp:nvxpyp),epstil(-lnxpyp:nvxpyp),
     2 tken(-lnxpyp:nvxpyp),epsn(-lnxpyp:nvxpyp)
c
      dimension spm(-lnxpyp:nvxpyp,lnsp)
C
C
      common /lekDATA/ ek(51,lnsp),eliq(51),pvap(lvap),visliq(lvap)
C
C
      equivalence (ek,hk),(eliq,hlat0)
      dimension hk(51,lnsp),hlat0(51)
C
      common /sc1/ aasc1(1),
     & adia,airdif,airla1,airla2,airmu1,airmu2,anc4o2,
     1 angmom,anu0,arrow,atdc,a0me,a0mom,botcyl,botfac,botin,b0,
     2 cafilm,cafin,cart3d,cdump,ces,ce1,ce13,ce2,ce3,cfilm,
     3 cfocbck,cmu,cmueps,conrod,cosect,cps,
     4 crank,csubk,csubmu,cyl,cylny,c1,dcadmp,distamb,dprefin,dt,dtacc,
     5 dtcad,dtcon,dtcyc0,dtfa,dtfc,dtfst,dtgrof,dtmax,dtmin,
     6 dtmxca,dtnm1,dto4,dtpgs,dtwp,dy,dzchop,d1,epschm,epse,epsk,
     7 epsm,epsp,epst,epsv,epsy,expdif,facsec,fchsp,
     8 flbowl,fldome,flface,flhead,flsqsh,fnfluxs,forthd,freslp,
     9 grfac,grind,gx,gy,gz,
     & zzsc1
C
      common /sc1i/ iaasc1(1),
     & ibigit,icart3,icont(26),icyl,
     1 iemax(lp),iho(lp),iignl(2),iignr(2),ijkall,ijkvec,
     2 ipo(lp),ipost,irec,irest,irez,jignd(2),jignf(2),jsectr,jterm,
     3 jtest,kho(lp),kht,kignb(2),kignt(2),kpo(lp),kptop,
     4 kwikeq,lpr,lwall,name(10),ncfilm,nchop,nclast,ncorr(lp),
     5 nctap8,ncyc,ndump,neo,nfluxs,nho,np,npn,npo,nre,nrk,nsp,nstrt,
     6 numcit,numeit,numkit,numpit,numvit,numyit,nunif,nx,nxp,
     7 nxpny,nxpnym,nxpnyp,ny,nyeq1,nynew,nyp,nz,nzp,
     8 nzzsc1
C
      common /sc1r/ aasc1r(1),
     & offset,omcyl,omgchm,omjsec,omnyq1,pamb,pardon,
     1 perr,pgs,pgsrat,pgssw,phidmx,phimax,pi,pio180,pio2,pi2,pi4o3r,
     2 pm,prl,q0,ranb,rans,rc,
     x rcornr,rdt,rerf(21),rfny,rgamamb,rgamin,rgas,rho(lp),
     1 rnfluxs,roamb,roin0,rpgs2,rpm,rpo(lp),rpr,rpre,rprq,rps,rsc,
     2 rsclmx,rtfac,rtnoslp,rtout,sampl,sclamb,sclmx,semimj(lp),
     3 semimn(lp),sgsl,sixth,skihco(lp),skipco(lp),snsect,square,
     4 zzsc1r
C
      common /sc1b/ aasc1b(1),
     & stb,stm,stroke,swipro,swirl,t,tchem,tcrit,tcut,
     1 tcute,tcylwl,tempi,tevap,tfilm,thead,third,thsect,timlmt,
     2 tkeamb,tkei,tkelow,tkesw,tlimd,topfac,topout,
     3 tpistn,tvalve,twfilm,twfin,twlvth,twothd,t1ign,
     4 t2ign,uniscal,u0,u1,visrat,win,wpistn,xignit,
     5 x0,y0,zhbot,zhead,zho(lp),zpistn,zpo(lp),
     6 zzsc1b
C
      common /injreal/ aainj(1),
     & amp0(lnoz),anoz(lnoz),breakup,cone(lnoz),
     1 costxy(lnoz),costxz(lnoz),dcone(lnoz),drnoz(lnoz),
     2 dthnoz(lnoz),dtinj,dznoz(lnoz),eavec(lnoz,3),envec(lnoz,3),
     3 eovec(lnoz,3),evapp,oscil0(lnoz),pminj,pulse,rdinj(100),
     4 rhop,sintxy(lnoz),sintxz(lnoz),smr(lnoz),tdinj,tfmass(lvel),
     5 tiltxy(lnoz),tiltxz(lnoz),tm2inj,tnparc,tpi,tspmas,turb,
     6 t1inj,t2inj,velinj(lvel),xinj(lnoz),yinj(lnoz),zinj(lnoz),
     7 zzinj
C
      common /injintg/ iaainj(1),
     1 injdist,kolide,numnoz,numvel,
     2 nzzinj
C
      common /chars/ idcon,iddt,idsp,jnm
C
C
C +++ IL NE FAUT SURTOUT PAS INITIALISER DANS BEGIN LES VALEURS EN DATA  !
C +++ NE PAS DUMPER LES DATA  (VALEUR INITIALISEE A LA COMPILATION =>
C +++                           PROBLEME SI IL Y A EQUIVALENCE )
C
C
      COMMON /MEXT1/ aaa10(1),
     &               zxl(lnpt1),beta1(lnpt1),beta2(lnpt2),ppol(lnpt2),
     1               ropol(lnpt2),upol(lnpt2),spdinl(lnsp),
     2               xlm1(nv),ylm1(nv),zlm1(nv),
     4               xlm2(nv),ylm2(nv),zlm2(nv),
     6               fvlu1(nv),fvlu2(nv),skico1(nv),
     7               ucrkin(lnpdb),qfr(nv),
     8               zzz10
C
      COMMON /MEXT2/ iii1(1),
     1               iupco(nv),idnco(nv),jupco(nv),jdnco(nv),iedge(nv),
     3               nnn1
C
      COMMON /INDATA/ flsoup,fldeb
C
      COMMON /MEXT3/ aaa11(1),
     &               rayon,squish,aoa,rfa,usoup,vsoup,wsoup,zxs,
     1               zxjeu,zxlmax,zxlmin,rich,remp,xmw,xair,xcar,rho0,
     2               fo2,fn2,scalin,uxkiva,uykiva,uzkiva,zptop1,rokiva,
     3               pkiva,tkiva,tkein,epsin,siein,deq1,deq2,xlpip,
     4               xcp,ycp,zcp,tumble,cebu,tmcar0,tmo20,tmn20,
     5               tgene,rogene,pgene,rosin,hispge,
     +               umoy,uin,vin,uvwin,bolh,sigma0,
     6               zzz11
C
      COMMON /MEXT4/ iii2(1),
     &               mshext,nkb,nkc,nksqsh,nks,kptop2,nsoup,npt1,npt2,
     1               nkdeb,jsep,npdb1,npdb2,kignb1,kignt1,kignb2,kignt2,
     2               jsep2,
     3               nnn2
C
      COMMON /LOGIC/ cropen,vopen
C
      real mb,mc,mf,ml,mp,mv,mvp,mw
      integer am,an,bm,bn,cm,cn
      character *8 idcon(26),idsp(15),jnm
      character *4 iddt



c      INCLUDE '../include/complt.h'
      common /pltintg/ IAAPLT(1),
     & iedg(lv),iface(lv*6),islc(lv),islv(lv),jslc(lv),
     1 jslv(lv),kslc(lv),kslv(lv),mirror,nvcont,nvpvec,nvvvec,nvzone,
     2 NZZPLT
C
      common /pltreal/ AAPLT(1),
     & cosal,cosphc(lv),cosphv(lv),cosphz(lv),
     1 costhc(lv),costhv(lv),costhz(lv),sinal,sinphc(lv),sinphv(lv),
     2 sinphz(lv),sinthc(lv),sinthv(lv),sinthz(lv),xbec(lv),xbev(lv),
     3 xbez(lv),xcplot,xez(lv),ybec(lv),ybev(lv),ybez(lv),ycplot,
     4 yez(lv),zbec(lv),zbev(lv),zbez(lv),zcplot,zez(lv),zz

      dimension khosav(lp)
      character *8 id(4)
      character *4 mess
C
c +++
c +++ hk arrays are the enthalpies of the species, taken from the
c +++ janaf thermochemical tables.  intervals are t=100(n-1), and units
c +++ are kcal/mole (later get converted to sie in ergs/gm).
c +++ species #1 is assumed to be the fuel, and its hk is loaded by
c +++ subroutine fuel.  species #2-12 below are as follows:
c +++ 2=o2, 3=n2, 4=co2, 5=h20, 6=h, 7=h2, 8=o, 9=n, 10=oh,
c +++ 11=co, 12=no
c +++
c +++ -----------------------------------------------------
c +++
      data (hk(n,2),n=1,51) /-2.075,-1.381,-.685,.013,.724,1.455,2.21,2.
     1 988,3.786,4.6,5.427,6.266,7.114,7.971,8.835,9.706,10.583,11.465,1
     2 2.354,13.249,14.149,15.054,15.966,16.882,17.804,18.732,19.664,20.
     3 602,21.545,22.493,23.446,24.403,25.365,26.331,27.302,28.276,29.25
     4 4,30.236,31.221,32.209,33.201,34.196,35.193,36.193,37.196,38.201,
     5 39.208,40.218,41.229,42.242,43.257/
      data (hk(n,3),n=1,51) /-2.072,-1.379,-.683,.013,.71,1.413,2.125,2.
     1 853,3.596,4.355,5.129,5.917,6.718,7.529,8.35,9.179,10.015,10.858,
     2 11.707,12.56,13.418,14.28,15.146,16.015,16.886,17.761,18.638,19.5
     3 17,20.398,21.28,22.165,23.051,23.939,24.829,25.719,26.611,27.505,
     4 28.399,29.295,30.191,31.089,31.988,32.888,33.788,34.690,35.593,36
     5 .496,37.4,38.306,39.212,40.119/
      data (hk(n,4),n=1,51) /-2.238,-1.543,-.816,.016,.958,1.987,3.087,4
     1 .245,5.453,6.702,7.984,9.296,10.632,11.988,13.362,14.75,16.152,17
     2 .565,18.987,20.418,21.857,23.303,24.755,26.212,27.674,29.141,30.6
     3 13,32.088,33.567,35.049,36.535,38.024,39.515,41.01,42.507,44.006,
     4 45.508,47.012,48.518,50.027,51.538,53.051,54.566,56.082,57.601,59
     5 .122,60.644,62.169,63.695,65.223,66.753/
      data (hk(n,5),n=1,51) /-2.367,-1.581,-.784,.015,.825,1.654,2.509,3
     1 .39,4.3,5.24,6.209,7.21,8.24,9.298,10.384,11.495,12.63,13.787,14.
     2 964,16.16,17.373,18.602,19.846,21.103,22.372,23.653,24.945,26.246
     3 ,27.556,28.875,30.201,31.535,32.876,34.223,35.577,36.936,38.3,39.
     4 669,41.043,42.422,43.805,45.192,46.583,47.977,49.375,50.777,52.18
     5 1,53.589,55.,56.413,57.829/
      data (hk(n,6),n=1,51) /-1.481,-.984,-.488,.009,.506,1.003,1.5,1.99
     1 6,2.493,2.99,3.487,3.984,4.481,4.977,5.474,5.971,6.468,6.965,7.46
     2 1,7.958,8.455,8.952,9.449,9.945,10.442,10.939,11.436,11.933,12.43
     3 ,12.926,13.423,13.92,14.417,14.914,15.41,15.907,16.404,16.901,17.
     4 398,17.895,18.391,18.888,19.385,19.882,20.379,20.875,21.372,21.86
     5 9,22.366,22.863,23.359/
      data (hk(n,7),n=1,51) /-2.024,-1.265,-.662,.013,.707,1.406,2.106,2
     1 .808,3.514,4.226,4.944,5.67,6.404,7.148,7.902,8.668,9.446,10.233,
     2 11.03,11.836,12.651,13.475,14.307,15.146,15.993,16.848,17.708,18.
     3 575,19.448,20.326,21.21,22.098,22.992,23.891,24.794,25.703,26.616
     4 ,27.535,28.457,29.385,30.317,31.253,32.194,33.139,34.088,35.042,3
     5 5.999,36.961,37.926,38.895,39.868/
      data (hk(n,8),n=1,51) /-1.608,-1.08,-.523,.01,.528,1.038,1.544,2.0
     1 48,2.55,3.052,3.552,4.051,4.551,5.049,5.548,6.046,6.544,7.042,7.5
     2 40,8.038,8.536,9.034,9.532,10.029,10.527,11.026,11.524,12.023,12.
     3 522,13.022,13.522,14.023,14.524,15.026,15.529,16.033,16.537,17.04
     4 3,17.549,18.057,18.565,19.075,19.586,20.098,20.611,21.126,21.641,
     5 22.158,22.676,23.195,23.715/
      data (hk(n,9),n=1,51) /-1.481,-.984,-.488,.009,.506,1.003,1.5,1.99
     1 6,2.493,2.99,3.487,3.984,4.481,4.977,5.474,5.971,6.468,6.965,7.46
     2 1,7.958,8.455,8.952,9.449,9.946,10.444,10.941,11.439,11.938,12.43
     3 7,12.936,13.437,13.939,14.441,14.946,15.451,15.959,16.469,16.98,1
     4 7.495,18.012,18.531,19.054,19.58,20.11,20.643,21.18,21.721,22.266
     5 ,22.816,23.37,23.928/
      data (hk(n,10),n=1,51) /-2.192,-1.467,-.711,.013,.725,1.432,2.137,
     1 2.845,3.556,4.275,5.003,5.742,6.491,7.252,8.023,8.805,9.596,10.39
     2 7,11.207,12.024,12.849,13.681,14.52,15.364,16.214,17.069,17.929,1
     3 8.794,19.662,20.535,21.411,22.291,23.174,24.06,24.949,25.841,26.7
     4 35,27.632,28.532,29.434,30.338,31.245,32.153,33.064,33.976,34.89,
     5 35.807,36.725,37.644,38.566,39.489/
      data (hk(n,11),n=1,51) /-2.072,-1.379,-.683,.013,.711,1.417,2.137,
     1 2.873,3.627,4.397,5.183,5.983,6.794,7.616,8.446,9.285,10.13,10.98
     2 ,11.836,12.697,13.561,14.43,15.301,16.175,17.052,17.931,18.813,19
     3 .696,20.582,21.469,22.357,23.248,24.139,25.032,25.927,26.822,27.7
     4 19,28.617,29.516,30.416,31.316,32.218,33.121,34.025,34.93,35.835,
     5 36.741,37.649,38.557,39.465,40.375/
      data (hk(n,12),n=1,51) /-2.197,-1.451,-.705,.013,.727,1.448,2.186,
     1 2.942,3.716,4.507,5.313,6.131,6.96,7.798,8.644,9.496,10.354,11.21
     2 7,12.084,12.955,13.829,14.706,15.587,16.469,17.534,18.241,19.129,
     3 20.02,20.911,21.805,22.7,23.596,24.493,25.392,26.291,27.192,28.09
     4 4,28.997,29.9,30.805,31.71,32.616,33.523,34.431,35.34,36.249,37.1
     5 59,38.07,38.982,39.894,40.807/
c +++
c +++ -----------------------------------------------------
c +++
      data pi /3.14159 26535 89793 23846/
      data rgas /8.3143e+7/
      data rc /122.0/
C
      data t,dtgrof,grind,tchem,tevap /0.0,1.02,0.0,1.0e-10,1.0e-10/
C
      data pgsrat /0.04/
      data dtfc,dtfa,dtfst /0.2,0.5,0.5/
      data flface,flbowl,flsqsh,fldome,flhead /1.0,2.0,3.0,4.0,5.0/
C
      DATA FLSOUP,FLDEB /6.0,7.0/
C
      data ce1,ce2,ce3,ces,cmu,cps /1.44,1.92,-1.0,1.50,.09,.16432/
      data tkelow / 1.0e-08 /
      data csubmu,csubk,cfocb / 10.0,8.0,0.666666666666667 /
      data tm2inj,tcrit /2*0.0/
      data ncyc,ndump,np /0,0,0/
      data ranb,rans /2396745.,2396745./
c
c <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
c
c +++ read data deck, compute derived and scalar quantities
c +++
c +++
c +++ if chopper is in use, current nz has been read in from
c +++ dump, and supersedes the value on itape; this is done
c +++ after this main block of reads.
c +++
      if(irest.gt.0) then
        nyold =ny
        nzsave=nz
      endif
c +++
c +++ read flag for optional post-processor dumps, where:
c +++ 0 = no post-processor dump file desired,
c +++ 1 = start post-processor file tape9 with this run, and
c +++ 2 = continue dumping on previous tape9, which must exist in
c +++     local file space:
c +++
      read ( 5,600) id(1),ipost
      if(irest.eq.0) ipost=min0(ipost,1)
      write(12,600) id(1),ipost
      if(ipost.eq.2) then
c        call insist('tape9',irep9)
c        if(irep9.ne.(-1)) call exit
      endif
      read ( 5,600) id(1),nx,id(2),ny,id(3),nz
      write(12,600) id(1),nx,id(2),ny,id(3),nz
C
        NZLU=NZ
CMODIF1
      READ ( 5,600) ID(1),MSHEXT
      WRITE(12,600) ID(1),MSHEXT
      IF(MSHEXT.EQ.1) THEN
        WRITE(59,*)'ON LIRA DONC UN MAILLAGE EXTERIEUR'
        WRITE(12,*)'ON LIRA DONC UN MAILLAGE EXTERIEUR'
      ENDIF
CMODIF2
      read ( 5,600) id(1),lwall,id(2),nchop,id(3),lpr,id(4),jsectr
      write(12,600) id(1),lwall,id(2),nchop,id(3),lpr,id(4),jsectr
      read ( 5,600) id(1),irez,id(2),ncfilm,id(3),nctap8,id(4),nclast
      write(12,600) id(1),irez,id(2),ncfilm,id(3),nctap8,id(4),nclast
      read ( 5,610) id(1),cafilm,id(2),cafin,id(3),cadump,id(4),dcadmp
      write(12,620) id(1),cafilm,id(2),cafin,id(3),cadump,id(4),dcadmp
      read ( 5,610) id(1),angmom,id(2),cyl,id(3),dy,id(4),pgssw
CMODIF1
      IF(MSHEXT.EQ.1) CYL=0.
CMODIF2
      write(12,620) id(1),angmom,id(2),cyl,id(3),dy,id(4),pgssw
      read ( 5,610) id(1),sampl,id(2),dti,id(3),dtmxca
      write(12,620) id(1),sampl,id(2),dti,id(3),dtmxca
      read ( 5,610) id(1),dtmax,id(2),tlimd,id(3),twfilm,id(4),twfin
      write(12,620) id(1),dtmax,id(2),tlimd,id(3),twfilm,id(4),twfin
      read ( 5,610) id(1),fchsp,id(2),stroke,id(3),squish,id(4),rpm
      write(12,620) id(1),fchsp,id(2),stroke,id(3),squish,id(4),rpm
      read ( 5,610) id(1),atdc,id(2),conrod,id(3),offset,id(4),swirl
      write(12,620) id(1),atdc,id(2),conrod,id(3),offset,id(4),swirl
      read ( 5,610) id(1),swipro,id(2),thsect,id(3),epsy
      write(12,620) id(1),swipro,id(2),thsect,id(3),epsy
      read ( 5,610) id(1),epsv,id(2),epsp,id(3),epst,id(4),epsk
      write(12,620) id(1),epsv,id(2),epsp,id(3),epst,id(4),epsk
      read ( 5,610) id(1),epse,id(2),gx,id(3),gy,id(4),gz
      write(12,620) id(1),epse,id(2),gx,id(3),gy,id(4),gz
      read ( 5,610) id(1),tcylwl,id(2),thead,id(3),tpistn,id(4),tvalve
      write(12,620) id(1),tcylwl,id(2),thead,id(3),tpistn,id(4),tvalve
      read ( 5,610) id(1),tempi,id(2),pardon,id(3),a0,id(4),b0
      write(12,620) id(1),tempi,id(2),pardon,id(3),a0,id(4),b0
      read ( 5,610) id(1),anc4,id(2),adia
      write(12,620) id(1),anc4,id(2),adia
      read ( 5,610) id(1),anu0,id(2),visrat,id(3),tcut
      write(12,620) id(1),anu0,id(2),visrat,id(3),tcut
      read ( 5,610) id(1),tcute,id(2),epschm,id(3),omgchm,id(4),tkei
      write(12,620) id(1),tcute,id(2),epschm,id(3),omgchm,id(4),tkei
      read ( 5,610) id(1),tkesw,id(2),sgsl,id(3),uniscal
      write(12,620) id(1),tkesw,id(2),sgsl,id(3),uniscal
      read ( 5,610) id(1),airmu1,id(2),airmu2,id(3),airla1,id(4),airla2
      write(12,620) id(1),airmu1,id(2),airmu2,id(3),airla1,id(4),airla2
      read ( 5,610) id(1),expdif,id(2),prl,id(3),rpr,id(4),rprq
      write(12,620) id(1),expdif,id(2),prl,id(3),rpr,id(4),rprq
      read ( 5,610) id(1),rpre,id(2),rsc,id(3),xignit
      write(12,620) id(1),rpre,id(2),rsc,id(3),xignit
C
        READ(  5,610)  ID(1),CEBU
        WRITE(12,620)  ID(1),CEBU
C
      read ( 5,610) id(1),t1ign,id(2),tdign,id(3),ca1ign,id(4),cadign
      write(12,620) id(1),t1ign,id(2),tdign,id(3),ca1ign,id(4),cadign
      read ( 5,600) id(1),iignl(1),id(2),iignr(1),id(3),jignf(1)
      write(12,600) id(1),iignl(1),id(2),iignr(1),id(3),jignf(1)
      read ( 5,600) id(1),jignd(1),id(2),kignb(1),id(3),kignt(1)
      write(12,600) id(1),jignd(1),id(2),kignb(1),id(3),kignt(1)
      read ( 5,600) id(1),iignl(2),id(2),iignr(2),id(3),jignf(2)
      write(12,600) id(1),iignl(2),id(2),iignr(2),id(3),jignf(2)
      read ( 5,600) id(1),jignd(2),id(2),kignb(2),id(3),kignt(2)
      write(12,600) id(1),jignd(2),id(2),kignb(2),id(3),kignt(2)
      read ( 5,600) id(1),kwikeq
      write(12,600) id(1),kwikeq
c      if(sampl.eq.1.0) call sample('tfile',4)
c      if(sampl.eq.1.0) call sampon
c +++
c +++ if ny differs on input file, convert 2-d cylindrical mesh
c +++ to 3-d sector or full circle mesh
c +++
      if(irest.gt.0) then
        nz=nzsave
        if(ny.ne.nyold) then
          nynew=ny
          ny=nyold
c          call tran3d
        endif
      endif
      third=1./3.
      twothd=2.*third
      forthd=4.*third
      sixth=0.5*third
      twlvth=0.5*sixth
      pi2=pi*2.0
      pio2=pi*0.5
      pio180=pi/180.
      arrow=tan(30.0*pio180)
      angmom=angmom*cyl
c      icart3=icvmgt(1,0,cyl.eq.0.0 .and. ny.gt.1)
      cart3d=float(icart3)
c      icyl=icvmgz(0,1,cyl)
c      cylny=cvmgt(1.0,0.0,cyl.eq.1.0 .and. ny.gt.1)
      omcyl=1.0-cyl
c      nyeq1=icvmgt(1,0,ny.eq.1)
      omnyq1=1.-float(nyeq1)
      if(jsectr.eq.1 .and. ny.gt.1 .and. amod(360.0,thsect).ne.0.0) then
        write(59,'(a)') ' thsect is not an even fraction of 360 degrees'
c        call exita(5)
      endif
      thread=thsect
      if(cyl.eq.0.) jsectr=0
      if(cyl.eq.1.0 .and. ny.eq.1) jsectr=1
c      thsect=cvmgzi(360.,thsect,jsectr)
c      thsect=cvmgzi(thsect,0.500,nyeq1)*pio180
      cosect=cos(thsect)
      snsect=sin(thsect)
      omjsec=1.-float(jsectr)
      jtest=max0(jsectr,nyeq1)
      offset=offset*omnyq1*omjsec
c      den=cvmgz(1.0,thsect,thsect)
c      facsec=cvmgz(1.0,pi2/den,cyl)
      nxp=nx+1
      nyp=ny+1
      nzp=nz+1
      nxpny=nxp*ny
      nxpnyp=nxp*nyp
      ijkall=nxpnyp*nzp
      if(ijkall.gt.nv .or. nxpnyp.gt.lnxpyp .or. nzp.gt.lnzp) then
        write(59,'(a)') ' parameter error: check nv, lnxpyp, and lnzp'
c        call exita(5)
      endif
      jterm=nxp*(1-nyeq1)
      nxpnym=nxp*(ny-1)
      kht=nz*nxpnyp
      ijkvec=kht-nxp-1
      rps=rpm/60.
      if(tkesw.eq.1.0 .and. lwall.ne.+1) then
        write(59,'(a)') ' resetting lwall=+1, because tke switch is on'
        lwall=+1
      endif
c      freslp=cvmgpi(1.0,0.0,lwall)
      if(tkesw.eq.1.0) visrat=-twothd
c      tkei=cvmgz(tkei,tkei*2.0*(stroke*rps)**2,rpm)
      tkei=tkei*tkesw
      phimax=amax1(rpr,rsc,2.+visrat,tkesw*rprq,tkesw*rpre)
      cmueps=sqrt(cmu*rpre/(ce2-ce1))
      ce13=twothd*ce1-ce3
      u0=0.875/sqrt(sqrt(cmu)*(ce2-ce1)/rpre)
      u1=sqrt(rc)-u0*alog(rc)
      q0=(prl*rpr-1.0)*sqrt(rc)
      a0me=a0
      a0mom=a0
      anc4o2=anc4*0.5
      if(t.eq.0.0 .or. pgssw.eq.0.0) pgs=1.0
      rpgs2=1./pgs**2
      grfac=1000./float(nx*ny*nz)
      cfocbck=cfocb/csubk
      x0=offset
      y0=0.
      dtcad=0.0
c      dtmxca=cvmgz(9.9e+9,dtmxca,rpm)
      if(rps.gt.0.0) then
        rrps36=1./(rps*360.)
        dtcad=rrps36
        dtmxca=dtmxca*dtcad
      endif
      if(t1ign.lt.0.0 .and. rpm.gt.0.0) then
        t1ign=abs(atdc-ca1ign)*rrps36
        tdign=abs(cadign)*rrps36
      endif
      t2ign=t1ign+tdign
      if(t.eq.0.0) then
        dt=dti
        cdump=cadump
        cfilm=atdc+cafilm
        tfilm=twfilm
      endif
c +++
c +++ read injector data
c +++
      read ( 5,600) id(1),numnoz,id(2),numvel,id(3),injdist,id(4),kolide
      write(12,600) id(1),numnoz,id(2),numvel,id(3),injdist,id(4),kolide
      read ( 5,610) id(1),t1inj,id(2),tdinj,id(3),ca1inj,id(4),cadinj
      write(12,620) id(1),t1inj,id(2),tdinj,id(3),ca1inj,id(4),cadinj
      read ( 5,610) id(1),tspmas,id(2),pulse,id(3),tnparc,id(4),rhop
      write(12,620) id(1),tspmas,id(2),pulse,id(3),tnparc,id(4),rhop
      read ( 5,610) id(1),tpi,id(2),turb,id(3),breakup,id(4),evapp
      write(12,620) id(1),tpi,id(2),turb,id(3),breakup,id(4),evapp
      pi4o3r=pi*forthd*rhop
      if(numnoz.eq.0) go to 25
      if(numnoz.gt.lnoz) then
        write(59,'(a)') ' parameter error:  numnoz > lnoz'
c        call exita(5)
      endif
      if(numvel.gt.lvel) then
        write(59,'(a)') ' parameter error:  numvel > lvel'
c        call exita(5)
      endif
      if(tnparc.gt.npar .and. pulse.gt.0.0)
     &  write(59,'(a)') ' warning:  tnparc > npar for pulsed spray'
      atotal=0.0
      do 5 i=1,numnoz
      read ( 5,610) id(1),drnoz(i),id(2),dznoz(i),id(3),dthnoz(i)
      write(12,620) id(1),drnoz(i),id(2),dznoz(i),id(3),dthnoz(i)
      read ( 5,610) id(1),tiltxy(i),id(2),tiltxz(i),id(3),cone(i),
     &              id(4),dcone(i)
      write(12,620) id(1),tiltxy(i),id(2),tiltxz(i),id(3),cone(i),
     &              id(4),dcone(i)
      read ( 5,610) id(1),anoz(i),id(2),smr(i),id(3),amp0(i)
      write(12,620) id(1),anoz(i),id(2),smr(i),id(3),amp0(i)
      dthnoz(i)=dthnoz(i)*pio180
      tiltxy(i)=tiltxy(i)*pio180
      tiltxz(i)=tiltxz(i)*pio180
      cone(i)  =cone(i)  *pio180
      dcone(i) =dcone(i) *pio180
      atotal=atotal+anoz(i)
    5 continue
      if(ny.eq.1) then
        numnoz=1
        dthnoz(1)=0.5*thsect
        tiltxy(1)=0.5*thsect
        if(cone(1).ne.dcone(1)) then
          tiltxz(1)=0.5*cone(1)
          cone(1)=dcone(1)
        endif
        anoz(1)=anoz(1)/facsec
        atotal=anoz(1)
      endif
      read ( 5,*) (velinj(i),i=1,numvel)
      if(t1inj.lt.0.0 .and. rpm.gt.0.0) then
        t1inj=abs(atdc-ca1inj) *rrps36
        tdinj=abs(cadinj)*rrps36
      endif
      t2inj=t1inj+tdinj
      if(pulse.lt.3.0) then
        tspmas=tspmas*anoz(1)/(atotal*facsec)
c        pminj=cvmgz(1.0,tnparc,tnparc)
        pminj=tspmas/pminj
      else
c +++
c +++ calculate total fuel mass predicted by the velocity table,
c +++ and correct the velocities by the ratio of mass desired to
c +++ mass predicted:
c +++
        dtinj=tdinj/float(numvel-1)
        arodt=atotal*rhop*dtinj*facsec
        fmpred=0.0
        do 10 i=2,numvel
        fmpred=fmpred+arodt*0.5*(velinj(i)+velinj(i-1))
   10   continue
        corect=tspmas/fmpred
        velinj(1)=velinj(1)*corect
        tfmass(1)=0.0
        arodt=anoz(1)*rhop*dtinj
        do 15 i=2,numvel
        velinj(i)=velinj(i)*corect
        tfmass(i)=tfmass(i-1)+arodt*0.5*(velinj(i)+velinj(i-1))
   15   continue
        pminj=tfmass(numvel)/tnparc
      endif
      write(12,880) (i,velinj(i),i=1,numvel)
c +++
c +++ compute 3 unit vectors (a, n, & o) for each nozzle, for use
c +++ in setting velocities in subroutine inject, where a is
c +++ along the axis of injection, n is the normal to a, and o is
c +++ other, given by the cross product of a and n:
c +++
      do 20 i=1,numnoz
      costxy(i)=cos(tiltxy(i))
      sintxy(i)=sin(tiltxy(i))
      costxz(i)=cos(tiltxz(i))
      sintxz(i)=sin(tiltxz(i))
      eavec(i,1)=+sintxz(i)*costxy(i)
      eavec(i,2)=+sintxz(i)*sintxy(i)
      eavec(i,3)=-costxz(i)
      envec(i,1)=+sin(tiltxz(i)+pio2)*costxy(i)
      envec(i,2)=+sin(tiltxz(i)+pio2)*sintxy(i)
      envec(i,3)=-cos(tiltxz(i)+pio2)
      eovec(i,1)=eavec(i,2)*envec(i,3) - eavec(i,3)*envec(i,2)
      eovec(i,2)=eavec(i,3)*envec(i,1) - eavec(i,1)*envec(i,3)
      eovec(i,3)=eavec(i,1)*envec(i,2) - eavec(i,2)*envec(i,1)
   20 continue
   25 continue
c +++
c +++ read data defining piston outline
c +++
      read ( 5,600) id(1),npo,id(2),nunif
      write(12,630) npo,nunif
CMODIF1
      if(npo.gt.lp.AND.MSHEXT.EQ.0) then
        write(59,'(a)') ' parameter error: npo > lp'
c        call exita(5)
      elseif(npo.eq.0.AND.MSHEXT.EQ.0) then
CMODIF2
        write(59,'(a)') ' cannot have npo = 0'
c        call exita(5)
      endif
      do 30 n=1,npo
      read ( 5,640) ipo(n),kpo(n),rpo(n),zpo(n)
   30 write(12,640) ipo(n),kpo(n),rpo(n),zpo(n)
C
C      if(sgsl.eq.0.0) sgsl=rpo(npo)
C      sclmx=sgsl/cmueps
C      rsclmx=1.0/sclmx
C
c +++
c +++ skipco table indicates sharp corners, which we define as a corner
c +++ with less than a 125 degree included angle measured through the
c +++ obstacle.  these  points are skipped (left unchanged) in the
c +++ general freeslip velocity treatment in subroutine bc, and are
c +++ flagged by setting skipco(n) = 0.0 in the test below.  on the
c +++ other hand, if the corner is acute (included angle measured
c +++ through the obstacle is > 235 degrees), skipco(n) = -1.0 to
c +++ indicate a separate corner treatment in subroutine bc.
c +++
C +++
C +++ LES DEFINITIONS DE SKIPCO ET DE SKIHCO DANS LE CAS D'UN
C +++ MAILLAGE LU EST DIFFERE JUSQU'A L'EXECUTION DE SETUP
C +++
C +++ LA MEME CHOSE POUR LA DEFINITION DE KPTOP, ZHEAD, ZBOT ETC....
C +++ ET TOUTE AUTRE QUANTITE GEOMETRIQUE RELATIVE AU BOL ET CAVITE.
C +++
C +++
C +++ SKIP IF EXTERNAL MESH
C +++
      IF(MSHEXT.EQ.0) THEN
C
      if(sgsl.eq.0.0) sgsl=rpo(npo)
      sclmx=sgsl/cmueps
      rsclmx=1.0/sclmx
C
      skipco(1)=1.0
      skipco(npo)=1.0
      flat=1.0
      do 40 n=2,npo-1
c      flat=cvmgt(0.0,flat,zpo(n).ne.zpo(1))
      nf=n+1
      nb=n-1
      drf=rpo(nf)-rpo(n)
      dzf=zpo(nf)-zpo(n)
      drb=rpo(nb)-rpo(n)
      dzb=zpo(nb)-zpo(n)
      dfdb=sqrt((drf**2+dzf**2)*(drb**2+dzb**2))
      cosine=(drf*drb+dzf*dzb)/dfdb
      sine=drb*dzf-drf*dzb
c      skipco(n)=cvmgt(0.0,1.0,cosine.gt.-0.573576 .and. sine.gt.0.0)
c      skipco(n)=cvmgt(-1.0,skipco(n),
c     &                cosine.gt.-0.573576 .and. sine.lt.0.0)
   40 continue
      zhead=stroke+zpo(npo)+squish
      zhbot=zhead
      kptop=kpo(npo)
c      nchop=icvmgz(0,nchop,rpm)
c      nchop=cvmgt(0,nchop,nchop.ge.nz)
c      nchop=cvmgt(2,nchop,nchop.eq.1)
      if(nchop.eq.0) go to 50
c +++
c +++ alternatively, to delay chopping in bowl geometry, set
c +++ dzchop = 0.999*squish / float(nchop)
c +++
c      dzchop=cvmgz(float(nchop+1),0.999,flat) * squish/float(nchop)
      write(12,770) dzchop
      write(59,770) dzchop
c +++
c +++ read data defining head outline
c +++
CMODIF1
      ENDIF
CMODIF2
   50 read ( 5,600) id(1),nho
      write(12,600) id(1),nho
CMODIF1
      if(nho.gt.lp.AND.MSHEXT.EQ.0) then
        write(59,'(a)') ' parameter error:  nho > lp'
c        call exita(5)
      endif
      IF(MSHEXT.EQ.1) GO TO 90
      if(nho.eq.0.AND.MSHEXT.EQ.0) go to 90
CMODIF2       
      zmax=-1.0e+10
      do 60 n=1,nho
      if(irest.gt.0) khosav(n)=kho(n)
      read ( 5,640) iho(n),kho(n),rho(n),zho(n)
      write(12,640) iho(n),kho(n),rho(n),zho(n)
      if(irest.gt.0) kho(n)=khosav(n)
      zmax=amax1(zmax,zho(n))
   60 continue
CMODIF1
C +++
C +++ SKIP IF EXTERNAL MESH
C +++
      IF(MSHEXT.EQ.0) THEN
CMODIF2
      zhead=zhead+zmax
      skihco(1)=1.0
      skihco(nho)=1.0
      do 70 n=2,nho-1
      nf=n+1
      nb=n-1
      drf=rho(nf)-rho(n)
      dzf=zho(nf)-zho(n)
      drb=rho(nb)-rho(n)
      dzb=zho(nb)-zho(n)
      dfdb=sqrt((drf**2+dzf**2)*(drb**2+dzb**2))
      cosine=(drf*drb+dzf*dzb)/dfdb
      sine=drf*dzb-drb*dzf
c      skihco(n)=cvmgt(0.0,1.0,cosine.gt.-0.573576 .and. sine.gt.0.0)
c      skihco(n)=cvmgt(-1.0,skihco(n),
c     &                cosine.gt.-0.573576 .and. sine.lt.0.0)
   70 continue
CMODIF1
      ENDIF
CMODIF2
      read ( 5,600) id(1),neo
      write(12,600) id(1),neo
      if(neo.eq.0) go to 90
      do 80 n=1,neo
      read ( 5,640) ncorr(n),iemax(n),semimj(n),semimn(n)
      write(12,640) ncorr(n),iemax(n),semimj(n),semimn(n)
   80 continue
c +++
c +++ read and check data for the square bowl option
c +++
   90 read ( 5,610) id(1),square,id(2),rcornr
      write(12,620) id(1),square,id(2),rcornr
      read ( 5,600) id(1),nstrt
      write(12,600) id(1),nstrt
CMODIF1
C +++
C +++ SKIP IF EXTERNAL MESH
C +++
      IF(MSHEXT.EQ.0) THEN
CMODIF2
      if(square.eq.1.0) then
        if(offset.ne.0.0) then
          write(59,'(a)') ' offset square bowls not allowed'
c          call exita(5)
        elseif(rcornr.le.0.0) then
          write(59,'(a)') ' rcornr must be > 0.0 for square bowl'
c          call exita(5)
        elseif(nho.gt.0) then
          write(59,'(a)') ' nho must be 0 for square bowls'
c          call exita(5)
        elseif(cyl.eq.0.0) then
          write(59,'(a)') ' cyl must be 1.0 for square bowls'
c          call exita(5)
        elseif(ny-2*nstrt.lt.1) then
          write(59,'(a)') ' ny incompatible with nstrt for square bowl'
c          call exita(5)
        elseif(thread.ne.90.0 .or. jsectr.eq.0) then
          write(59,'(a)') ' square bowl requires 90-degree sector mesh'
c          call exita(5)
        endif
      endif
CMODIF1
      ENDIF
CMODIF2
c +++
c +++ read contour plot binary flags and
c +++ setup zone, velocity vector, and contour plots
c +++ ensure kslv & kslc choices within mesh, as chopper can affect them
c +++ (x-,y-,zcplot specialize plot center to engine geometry)
c +++
      read ( 5,'(a8,26i1)') id(1),(icont(n),n=1,26)
      write(12,'(a8,26i1)') id(1),(icont(n),n=1,26)
      xcplot=0.0
      ycplot=0.0
      zcplot=zhead-0.5*stroke
      read ( 5,600) id(1),mirror
c      mirror=cvmgt(mirror,0,ny.eq.1 .and. cyl.eq.1.0)
      write(12,600) id(1),mirror
      rfny=1.0/float(ny)
      thony=thsect*rfny
      cosal=cos(thony)
      sinal=sin(thony)
      omeps=1.-(1.e-10)
      read ( 5,600) id(1),nvzone
      write(12,650) nvzone
      if(nvzone.gt.lv) then
        write(59,'(a)') ' parameter error:  nvzone > lv'
c        call exita(5)
      endif
      if(nvzone.eq.0) go to 110
      do 100 n=1,nvzone
      if1=(n-1)*6+1
      if6=if1+5
      read ( 5,660) xez(n),yez(n),zez(n),(iface(i),i=if1,if6),iedg(n)
      write(12,670) xez(n),yez(n),zez(n),(iface(i),i=if1,if6),iedg(n)
      if(xez(n).eq.xcplot) xez(n)=xez(n)+1.e-6
      if(yez(n).eq.ycplot) yez(n)=yez(n)+1.e-6
      if(zez(n).eq.zcplot) zez(n)=zez(n)+1.e-6
      xdist=xez(n)-xcplot
      ydist=yez(n)-ycplot
      zdist=zez(n)-zcplot
      ground=amax1(sqrt(xdist**2+ydist**2),1.e-6)
      radeye=sqrt(zdist**2+ground**2)
      xog=xdist/ground
c +++
c +++ avoid asin of 1.0 + low-order bit, which can occur on cray
c +++
      if(abs(xog).gt.omeps) xog=sign(1.,xog)
      theta=asin(xog)
      yog=ydist/ground
      if(abs(yog).gt.omeps) yog=sign(1.,yog)
      q1=asin(yog)+pio2
      if(ydist.gt.0. .and. xdist.gt.0.) theta=q1
      if(ydist.gt.0. .and. xdist.lt.0.) theta=-q1
      zor=zdist/radeye
      if(abs(zor).gt.omeps) zor=sign(1.,zor)
      phi=asin(zor)
      sinthz(n)=sin(theta)
      costhz(n)=cos(theta)
      sinphz(n)=sin(phi)
      cosphz(n)=cos(phi)
      xbez(n)=xdist*costhz(n)+ydist*sinthz(n)
      yzterm=ydist*costhz(n)-xdist*sinthz(n)
      ybez(n)=cosphz(n)*yzterm-zdist*sinphz(n)
      zbez(n)=sinphz(n)*yzterm+zdist*cosphz(n)
c +++
c +++ ensure left, front, derriere faces not plotted in 3-d cyl.
c +++
c      inop=cvmgt(0,1,cyl.eq.1.0 .and. ny.gt.1 .and. jsectr.eq.0)
      ifx=(n-1)*6
      iface(ifx+1)=iface(ifx+1)*inop
      iface(ifx+3)=iface(ifx+3)*inop
      iface(ifx+4)=iface(ifx+4)*inop
  100 continue
  110 if(ny.eq.1) nvzone=1
      read ( 5,600) id(1),nvvvec,id(2),nvpvec
      write(12,680) nvvvec,nvpvec
      if(nvvvec.gt.lv) then
        write(59,'(a)') ' parameter error:  nvvvec > lv'
c        call exita(5)
      endif
      if(nvvvec.eq.0 .and. ny.gt.1) go to 150
c      nvvv=icvmgt(1,nvvvec,nvvvec.eq.0 .and. ny.eq.1)
      do 140 n=1,nvvv
      if(nvvvec.eq.0 .and. ny.eq.1) go to 120
      read ( 5,690) xev,yev,zev,islv(n),jslv(n),kslv(n)
      kslv(n)=min0(kslv(n),nz)
      write(12,695) xev,yev,zev,islv(n),jslv(n),kslv(n)
      if(ny.gt.1) go to 130
  120 xev=0.
      yev=-1.0e10
      zev=zcplot
      islv(n)=0
      jslv(n)=1
      kslv(n)=0
  130 if(xev.eq.xcplot) xev=xev+1.e-6
      if(yev.eq.ycplot) yev=yev+1.e-6
      if(zev.eq.zcplot) zev=zev+1.e-6
      xdist=xev-xcplot
      ydist=yev-ycplot
      zdist=zev-zcplot
      ground=amax1(sqrt(xdist**2+ydist**2),1.e-6)
      radeye=sqrt(zdist**2+ground**2)
      xog=xdist/ground
c +++
c +++ avoid asin of 1.0 + low-order bit, which can occur on cray
c +++
      if(abs(xog).gt.omeps) xog=sign(1.,xog)
      theta=asin(xog)
      yog=ydist/ground
      if(abs(yog).gt.omeps) yog=sign(1.,yog)
      q1=asin(yog)+pio2
      if(ydist.gt.0. .and. xdist.gt.0.) theta=q1
      if(ydist.gt.0. .and. xdist.lt.0.) theta=-q1
      zor=zdist/radeye
      if(abs(zor).gt.omeps) zor=sign(1.,zor)
      phi=asin(zor)
      sinthv(n)=sin(theta)
      costhv(n)=cos(theta)
      sinphv(n)=sin(phi)
      cosphv(n)=cos(phi)
      xbev(n)=xdist*costhv(n)+ydist*sinthv(n)
      yzterm=ydist*costhv(n)-xdist*sinthv(n)
      ybev(n)=cosphv(n)*yzterm-zdist*sinphv(n)
      zbev(n)=sinphv(n)*yzterm+zdist*cosphv(n)
  140 continue
      if(ny.eq.1) nvvvec=1
  150 read ( 5,600) id(1),nvcont
      write(12,700) nvcont
      if(nvcont.gt.lv) then
        write(59,'(a)') ' parameter error:  nvcont > lv'
c        call exita(5)
      endif
      if(nvcont.eq.0 .and. ny.gt.1) go to 190
c      nvco=icvmgt(1,nvcont,nvcont.eq.0 .and. ny.eq.1)
      do 180 n=1,nvco
      if(nvcont.eq.0 .and. ny.eq.1) go to 160
      read ( 5,690) xec,yec,zec,islc(n),jslc(n),kslc(n)
      kslc(n)=min0(kslc(n),nz)
      write(12,695) xec,yec,zec,islc(n),jslc(n),kslc(n)
      if(ny.gt.1) go to 170
  160 xec=0.
      yec=-1.0e10
      zec=zcplot
      islc(n)=0
      jslc(n)=1
      kslc(n)=0
  170 if(xec.eq.xcplot) xec=xec+1.e-6
      if(yec.eq.ycplot) yec=yec+1.e-6
      if(zec.eq.zcplot) zec=zec+1.e-6
      xdist=xec-xcplot
      ydist=yec-ycplot
      zdist=zec-zcplot
      ground=amax1(sqrt(xdist**2+ydist**2),1.e-6)
      radeye=sqrt(zdist**2+ground**2)
      xog=xdist/ground
c +++
c +++ avoid asin of 1.0 + low-order bit, which can occur on cray
c +++
      if(abs(xog).gt.omeps) xog=sign(1.,xog)
      theta=asin(xog)
      yog=ydist/ground
      if(abs(yog).gt.omeps) yog=sign(1.,yog)
      q1=asin(yog)+pio2
      if(ydist.gt.0. .and. xdist.gt.0.) theta=q1
      if(ydist.gt.0. .and. xdist.lt.0.) theta=-q1
      zor=zdist/radeye
      if(abs(zor).gt.omeps) zor=sign(1.,zor)
      phi=asin(zor)
      sinthc(n)=sin(theta)
      costhc(n)=cos(theta)
      sinphc(n)=sin(phi)
      cosphc(n)=cos(phi)
      xbec(n)=xdist*costhc(n)+ydist*sinthc(n)
      yzterm=ydist*costhc(n)-xdist*sinthc(n)
      ybec(n)=cosphc(n)*yzterm-zdist*sinphc(n)
      zbec(n)=sinphc(n)*yzterm+zdist*cosphc(n)
  180 continue
      if(ny.eq.1) nvcont=1
c +++
c +++ species data input:
c +++ species #1 is assumed to be the fuel, and subroutine
c +++ fuel is called to identify it from the formula
c +++ supplied below, and to assign all required data beyond
c +++ the initial density.  for the remaining species, the
c +++ formula, initial density, molecular weight, and heat
c +++ of formation are read in below.  htform is converted
c +++ from kcal/mole to cgs units for all the species.
c +++ copy species formulae to idcon table, for contour plot labels
c +++
  190 read ( 5,600) id(1),nsp
      write(12,710) id(1),nsp
      if(nsp.gt.lnsp .or. (lnsp.lt.3 .and. nchop.gt.0))then
        write(59,'(a)') ' nsp > lnsp, or lnsp < 3 and chopper on'
c        call exita(5)
      endif
      do 200 isp=1,nsp
      if(isp.eq.1) then
        read ( 5,720) idsp(1),id(1),rhoi(1)
c        call fuel
        id(2)=' mw1  '
        id(3)=' htf1 '
      else
        read ( 5,730) idsp(isp),id(1),rhoi(isp),id(2),mw(isp),
     1                          id(3),htform(isp)
      endif
      rmw(isp)=1./mw(isp)
      htform(isp)=htform(isp)*4.184e+10
      write(12,740) idsp(isp),id(1),rhoi(isp),id(2),mw(isp),
     1                        id(3),htform(isp)
      idcon(isp+11)=idsp(isp)
  200 continue
c +++
c +++ calculate initial droplet oscillation frequency at injector.
c +++ stinj is surface tension of injected fuel, visinj is its viscosity
c +++
      if(tpi.gt.tcrit .or. breakup.eq.0.0 .or. numnoz.eq.0) go to 210
      stinj=amax1(stm*tpi+stb,1.e-6)
      tb=tpi*0.1
      itb=int(tb)
      fr=tb-float(itb)
      visinj=fr*visliq(itb+2)+(1.0-fr)*visliq(itb+1)
      visinj=amax1(visinj,1.e-10)
      do 205 i=1,numnoz
      oscil0(i)=sqrt(csubk*stinj/(rhop*smr(i)**3)
     &          -(csubmu*visinj/(2.0*rhop*smr(i)**2))**2)
  205 continue
c +++
c +++ read in boundary condition data for right, top, bottom
c +++
  210 read ( 5,610) id(1),rtout,id(2),topout,id(3),botin
      write(12,620) id(1),rtout,id(2),topout,id(3),botin
      botfac=freslp*(1.-botin )
      rtfac =freslp*(1.-rtout )
      topfac=freslp*(1.-topout)
      botcyl=botin*cyl
c      rtnoslp=cvmgt(1.0,0.0,rtout.eq.0.0 .and. freslp.eq.0.0)
C
CAS STATIONNAIRE
        IF(MSHEXT.EQ.1.AND.RPM.EQ.0..AND.(BOTIN+TOPOUT).GT.0.0) THEN
          READ  (5,610) ID(1),DISTAMB,ID(2),PAMB,ID(3),SCLAMB
          WRITE(12,620) ID(1),DISTAMB,ID(2),PAMB,ID(3),SCLAMB
          SCLAMB=SCLAMB/CMUEPS
C +++     LES AUTRES VALEURS SONT DEFINIES DANS SETUP.
          GO TO 224
        ENDIF
C
      if(rtout+topout+botin .gt. 0.0) then
        read ( 5,610) id(1),distamb,id(2),pamb,id(3),tkeamb,id(4),sclamb
        write(12,620) id(1),distamb,id(2),pamb,id(3),tkeamb,id(4),sclamb
        sclamb=sclamb/cmueps
        do 215 isp=1,nsp
        read ( 5,610) id(1),spdamb(isp)
        write(12,620) id(1),spdamb(isp)
  215   continue
      endif
      if(botin.eq.1.0) then
        read ( 5,610) id(1),win
        write(12,620) id(1),win
          do 220 isp=1,nsp
          read ( 5,610) id(1),spdin0(isp)
          write(12,620) id(1),spdin0(isp)
  220     continue
      endif
c +++
c +++ read in kinetic reaction data
c +++
 224  CONTINUE
C
      read ( 5,600) id(1),nrk
      write(12,600) id(1),nrk
      if(nrk.eq.0) go to 250
      if(nrk.gt.lnrk) then
        write(59,'(a)') ' parameter error:  nrk > lnrk'
c        call exita(5)
      endif
      do 240 ir=1,nrk
      read ( 5,750) cf(ir),ef(ir),zetaf(ir)
      read ( 5,750) cb(ir),eb(ir),zetab(ir)
      read ( 5,780) (am(isp,ir),isp=1,nsp)
      read ( 5,780) (bm(isp,ir),isp=1,nsp)
      read ( 5,790) (ae(isp,ir),isp=1,nsp)
      read ( 5,790) (be(isp,ir),isp=1,nsp)
c +++
c +++ set reaction-species index arrays
c +++
      nk=0
      qr(ir)=0.
      do 230 isp=1,nsp
      if(am(isp,ir).eq.0 .and. bm(isp,ir).eq.0) go to 230
      nk=nk+1
      cm(nk,ir)=isp
      fam(isp,ir)=float(am(isp,ir))
      fbm(isp,ir)=float(bm(isp,ir))
      fbmam(isp,ir)=fbm(isp,ir)-fam(isp,ir)
      qr(ir)=qr(ir)-fbmam(isp,ir)*htform(isp)
  230 continue
      nelem(ir)=nk
      write(12,800) ir,qr(ir)
      write(12,810) cf(ir),ef(ir),zetaf(ir)
      write(12,820) cb(ir),eb(ir),zetab(ir)
      mess=' lhs'
      write(12,830) mess,(am(isp,ir),isp=1,nsp)
      mess=' rhs'
      write(12,830) mess,(bm(isp,ir),isp=1,nsp)
      mess=' for'
      write(12,840) mess,(ae(isp,ir),isp=1,nsp)
      mess='back'
      write(12,840) mess,(be(isp,ir),isp=1,nsp)
  240 continue
c +++
c +++ read in equilibrium reaction data only when the general
c +++ equilibrium solver is specified (kwikeq=0):
c +++
  250 if(kwikeq.eq.1) go to 280
      read ( 5,600) id(1),nre
      write(12,600) id(1),nre
      if(nre.eq.0) go to 280
      if(nre.gt.lnre) then
        write(59,'(a)') ' parameter error:  nre > lnre'
c        call exita(5)
      endif
      do 270 ire=1,nre
      read ( 5,760) as(ire),bs(ire),cs(ire),ds(ire),es(ire)
      read ( 5,780) (an(isp,ire),isp=1,nsp)
      read ( 5,780) (bn(isp,ire),isp=1,nsp)
      nk=0
      qeq(ire)=0.
      do 260 isp=1,nsp
      if(an(isp,ire).eq.0 .and. bn(isp,ire).eq.0) go to 260
      nk=nk+1
      cn(nk,ire)=isp
      fbnan(isp,ire)=float(bn(isp,ire))-float(an(isp,ire))
      qeq(ire)=qeq(ire)-fbnan(isp,ire)*htform(isp)
  260 continue
      nlm(ire)=nk
      write(12,850) ire,as(ire),bs(ire),cs(ire),ds(ire),es(ire)
      write(12,860) qeq(ire),nlm(ire)
      mess='an'
      write(12,870) mess,(an(isp,ire),isp=1,nsp)
      mess='bn'
      write(12,870) mess,(bn(isp,ire),isp=1,nsp)
      mess='cn'
      write(12,870) mess,(cn(isp,ire),isp=1,nsp)
  270 continue
c +++
c +++ convert hk arrays of enthalpy (in kcal/mole) to ek arrays
c +++ of sie (in ergs/gm).  1.987e-03 is universal gas constant
c +++ in kcal/mole degree kelvin; 4.184e+10 is ergs/kcal.  terms
c +++ in () represent sie in kcal/mole, with reference to zero
c +++ at absolute zero:
c +++
  280 do 290 isp=1,lnsp
      hkzero=hk(1,isp)
      do 290 n=1,51
      tempe=100.*float(n-1)
      ek(n,isp)=4.184e+10*rmw(isp)*(hk(n,isp)-hkzero-1.987e-03*tempe)
  290 continue
CMODIF1
C       DO 291 ISP=13,14
C       DO 291 N=1,51
C       EK(N,ISP)=EK(N,1)
C  291 CONTINUE
CMODIF2
c +++
c +++ convert latent heat of the liquid using these ek(n,1) values
c +++
      do 310 n=1,51
      tempe=100.*float(n-1)
      if(tempe.ge.tcrit) go to 300
      eliq(n)=ek(n,1)+rgas*tempe*rmw(1)-pvap(10*n-9)/rhop-hlat0(n)
      go to 310
  300 eliq(n)=ek(n,1)
  310 continue
c +++
c +++ create inverse error function table, used in subroutine
c +++ pmovtv for computing turbulent velocity components
c +++
      rerf(1)=0.
      rerf(21)=2.
      do 330 k=2,20
      err=0.05*float(k-1)
      xg=err*(1.+err)
  320 continue
c     fn=erf(xg)-err
      dfn=0.8862269255*exp(xg*xg)
      diff=-fn*dfn
      xg=xg+diff
      if(abs(diff).lt.1.e-07) go to 330
      go to 320
  330 rerf(k)=xg
c +++
c +++ create spray mass fraction table, used to calc. radp in inject
c +++
      xx=12.0
      rrd100=1.0/(1.0-exp(-xx)*(1.+xx+.5*xx**2+sixth*xx**3))
      do 340 n=1,100
      xx=0.12*float(n)
      rdinj(n)=(1.0-exp(-xx)*(1.+xx+.5*xx**2+sixth*xx**3))*rrd100
  340 continue
C
C
C       MODIF POUR RESTART:
C
        IF(MSHEXT.EQ.1.AND.IREST.NE.0) THEN
                NPO=1
                NHO=0
                IF(NKC.NE.0) NHO=1
                RPO(NPO)=RAYON
                KPO(NPO)=KPTOP
                KHO(NHO)=KPTOP2
                IF(SGSL.EQ.0.0) SGSL=RPO(NPO)
                SCLMX=SGSL/CMUEPS
                RSCLMX=1.0/SCLMX
C
                IF(NZLU.NE.NZSAVE) THEN
                  KIGNB(1)=KIGNB1
                  KIGNB(2)=KIGNB2
                  KIGNT(1)=KIGNT1
                  KIGNT(2)=KIGNT2
                ENDIF
C
                IF(CRANK.LT.CADUMP) CDUMP=CADUMP
C
                RHOI(1)=XCAR*RHO0
                RHOI(2)=XAIR*RHO0*FO2
                RHOI(3)=XAIR*RHO0*FN2
                RHOI(4)=0.0
                RHOI(5)=0.0
                OPEN(13,FILE='swirl.evol',FORM='FORMATTED')
                OPEN(14,FILE='tumbl.evol',FORM='FORMATTED')
                OPEN(10,FILE='brule.ebu', FORM='FORMATTED')
                OPEN(15,FILE='debit.ncyc',FORM='FORMATTED')
                WRITE(13,8007)
                WRITE(14,8006)
                WRITE(10,8012)
                WRITE(15,8015)
        ENDIF
C
C
      return
c
  600 format(a8,i5)
  610 format(a8,f10.5)
  620 format(a8,2x,1pe12.5)
  630 format(' npo=',i4,' nunif=',i4/' ipo kpo     rpo     zpo')
  640 format(2i4,2f8.5)
  650 format(/' nvzone=',i2/4x,'xez',8x,'yez',8x,'zez',
     & 5x,'l r f d b t iedg')
  660 format(3f8.2,7i2)
  670 format(3(1pe11.3),7i2)
  680 format(/' nvvvec=',i2,' nvpvec=',i2/4x,'xev',8x,'yev',8x,'zev',
     & 5x,'isl,jsl,ksl')
  690 format(3f8.2,3i4)
  695 format(3(1pe11.3),3i4)
  700 format(/' nvcont=',i2/4x,'xec',8x,'yec',8x,'zec',
     & 5x,'isl,jsl,ksl')
  710 format(//a8,i5/25x,'species data'//)
  720 format(a8,2x,a6,f10.5)
  730 format(a8,2x,3(a6,f10.5))
  740 format(a8,2x,3(a6,1pe14.5))
  750 format(3(6x,f10.5))
  760 format(5(6x,f10.5))
  770 format(' dzchop=',1pe12.5)
  780 format(5x,15i5)
  790 format(5x,8f8.3)
  800 format(//32x,'kinetic reaction',i5/' qr =' ,1pe20.6)
  810 format(' cf, ef, zetaf =',1p3e20.8)
  820 format(' cb, eb, zetab =',1p3e20.8)
  830 format(1x,a4,' stoichiometric coefs =',15i5/(26x,15i5))
  840 format(1x,a4,'ward species exponents =',8f8.3/(29x,8f8.3))
  850 format(//28x,'equilibrium reaction',i5/
     & ' as, bs, cs, ds, es =',1p5e18.8)
  860 format(' qeq =',1pe18.8,10x,'nlm =',i5)
  870 format(1x,a2,' =',15i5/(5x,15i5))
  880 format(' v(',i3,') = ',g13.5)
 8006 FORMAT(5X,'CRANK',7X,'TETAX')
 8007 FORMAT(5X,'CRANK',7X,'OMEGA')
 8012 FORMAT(5X,'CRANK',6X,'CARBURANT',9X,'O2',13X,'N2')
 8015 FORMAT(1X,'NCYC',4X,'INX',5X,'INY',5X,'INZ',5X,'OUTX',
     &  4X,'OUTY',4X,'OUTZ',4X,'TOTIN',3X,'TOTOUT',4X,'TIME')
C
      end
