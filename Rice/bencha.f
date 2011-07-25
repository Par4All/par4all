C     Modifications
C      - proper dimensions in common declarations of module BOUCLE
C      - removal of entry CNTRL in subroutine BOUCLE
C      - replacement of RETURN statements by END statements
C      - small commons, with only one array of one element, are
C     properly declared

      PROGRAM BENCHA
C
c
C-----EQUATION DE LA CHALEUR NON LINEAIRE 2 D
C-----DIFFERENCES FINIES, NEWTON, GRADIENT CONJUGUE
C
C-----VERSION 1.0 UTILISEE POUR LES TESTS CRAY X-MP, CRAY 2 
C-----21-28 AVRIL 1986
C
C-----COMPRESSIONS, RECHERCHES ET INTERPOLATIONS EN FORTRAN 
C-----NON VECTORIEL 
C
C-----ADRESSAGE INDIRECT
C
      PARAMETER (NNLIG=11,NNCOL=31,NTOD=(NNLIG+1)*NNCOL)
       COMMON/MONITOR/
     .LKOM(6),
     1IGENED,
     1 NUPROT,NOMCAS(8),T2003,NINTO,INVTOT,ARRET,MODIF(10),JDATV,NVERS,
     1 NUPRB13,NOMJOB,PGNOM,
     1 DATESG,NOMSAG,NUPRB11,IDATE,ICYCLE,
     1IGENEF,
     2IDIMD,
     2 NMIL,NTOT,NTOTM,NMF,NMAXNL,NFUS,NNTOT,NBNLOG,NBCOL,LIGDER,TRMG,
     2 TRMG2,NBGR,NCORPS,NREA,NSECNE,NNPOL,NNPOLM,NFUSM,
     2IDIMF
      COMMON/MONITOR/
     3LOGIQD,
     3 REPRISE,YYYY,MOVE,ZZZZ,MYYYYYO,MXXXXX,MZZZZZ,IOPACI,IAXE,ISCHEMA,
     3 ISYM,MINVER,ICOURB,SAGACE,IPRLCM,LOI4,LOI7,B13,
     3 VIDEC,COMBLE,PREC,INTNEU,LISS,DECOUPL,SORTZON,NEUDISK,OPADISK, 
     3 B10,DENEU,FUINE,PROTDIS,REPDIS,SANPROT,
     3SORMU,FUS,FUSD,LREMAIL,POLUT,LPRES,LENER,LABAN,
     3 B11,
     3 IFB10,IFRO10,IFEN10,
     3 SANSAG,
     3 METMCC,VARMAIL,
     3IANISO,
     3 IFPART,
     3 LLIBRE(5),LOGIQF
      COMMON /MONITOR/
     4IGESTD,
     4 LARGE(20),NBMOT(20),IADLCM,LCMLON,LON402,
     4L03,L04,L0402,L07,L0822,LETAT,LOPA,LSECT,LVARHE,LTOT,JOPAC,JETAT,
     4JSECT,LXXXPHO,IFLU,ICOU,IFUIT,IYYYY1,INTEG,INTLI,ISORTI,
     4LONFLU,LONCOU,LONFUIT,LONET,LSORTI,NEUDIS,LYYYY,
     4LONMC1,LONMC2,JTIMES,LTIMES,JBILAN,LBILAN,JTDEC,JVRDEC,JVZDEC,
     4 JRDEC,JZDEC,JZONES,JWS1DEC,JWS2DEC,JZMPDEC,
     4 IGESTF
      COMMON /MONITOR/
     5LCOM1,LCOM2,LCOM3,LCOM4,LCOM5,LCOM6,LCOM7,LCOM8,LCOM9,LCOM10,
     .LCREMAY,
     5LCOMFIN
      COMMON/DESCR/KN(NTOD)
     .,LN(NTOD),
     . MAT(NTOD),
     . ND(NTOD),
     . NH(NTOD),
     . NG(NTOD),
     . NB(NTOD),
     . KLFB(NTOD),
     . KLFG(NTOD),
     . IVID(NTOD),
     . DEN(NTOD),MNC,XT(20),XRO(40),XOPA(20,40),OPAK(NTOD), 
     . DP(NTOD),NPFG,NFG(NTOD),NPFB,NFB(NTOD),
     . V1(NTOD),V2(NTOD),V3(NTOD),V4(NTOD),V5(NTOD),V6(NTOD),V7(NTOD),
     . VOLU(NTOD),VOLUN(NTOD),DVOLU(NTOD),
     . NBLIG,NBKOL,NELIG(NTOD),NECOL(NTOD),IDLIG(NTOD),KOL(NTOD)
      COMMON/LICOL/NLIG,NCOL
      INTEGER DATESG
      LOGICAL ARRET 
       LOGICAL REPRISE,YYYY,MOVE,ZZZZ,SAGACE,LOI4,LOI7
      LOGICAL B13,SORTZON,NEUDISK,OPADISK
      LOGICAL B10,DENEU,FUINE,PROTDIS,REPDIS,SANPROT
      LOGICAL IFB10,IFRO10,IFEN10
      LOGICAL SANSAG
      LOGICAL VIDEC,COMBLE,PREC,INTNEU,LISS,DECOUPL,IPRLCM
      LOGICAL SORMU,FUS,FUSD,LREMAIL,POLUT,LPRES,LENER,LABAN
      LOGICAL IANISO
      LOGICAL B11,VARMAIL
      LOGICAL IFPART
      COMMON/SORTIES/
     *TPHASE(5,5),KDBS(5),LDBS(5),KFBS(5),LFBS(5),ITYPB(5,5),IFBS(5,5),
     *ICLIST(5),KFREQ(5,5),LFREQ(5,5),DTLIST(5,5),TLIST(5)
     *,NBLOCS
C  PROT
     *,TFPROT(5),DTPROT(5),TPROT,TMPROT,ITPROT,NPALP,IPROT,DTPSTD
C  BANDE 13
     *,KDB13,LDB13,KFB13,LFB13,NBPT13,VLISB13,TRELB13,TFAZB13(5)
     *,DTB13(5)
     *,TBECAN,TMAC,MAXITER,
     *ICNUS(5,5),
C  SAGACE 
     * NBSG,KDSG(5),LDSG(5),KFSG(5),LFSG(5),
     * NTSG,PTSG(5),DTSG(5),NVSG,NMSG,
     * DELTAZ,DELTAR,MIROIR,
     *             DERSOR
      LOGICAL MIROIR
      COMMON/TIMES/ 
     *             NITER,NITERH,NITERF,NITERN,
     *TEMPS,TIMEH,TIMEF,TIMEN,DTIT,DTH,DTF,DTN,DTHN,DTFN,DTNN,DTHS,DTFS,
     *DTNS,DTAH,DTAF,DTV,TARRET,DTITN,DTMIN,DTMAXF,DTMAXH
     *,TDEBFU,TDPSG 
     .,TDREM
     *,DERTIME
      COMMON/BILAN/ 
     *             ETOTO,ECINO,EINTO,ERADO,QMRO,QMZO,
     *             WEXT,QEXT,XIMPR,XIMPZ,XIMPA,
     *             EFIS,EFUS,EFMC,EMAT, 
     *             WQT1T,WQT2T,WQT3T,WQT,
     *             ETOT,ECIN,EINT,ERAD,QMR,QMZ,
     *             DWEXT,DQEXT,DXIMPR,DXIMPZ,DXIMPA,
     *             DEFIS,DEFUS,WQT1,WQT2,WQT3,
     *             TMA,XMA,VOL,
     *WNMIL(50),WFMIL(50),
     *             DERBIL
      COMMON/C0201/R(NTOD)
      COMMON/C0202/Z(NTOD)
      COMMON/C0203/RO(NTOD)
      COMMON/C0204/T(NTOD)
      COMMON/C0205/ZMM(NTOD)
      COMMON/C0206/DWA(NTOD)
      COMMON/C0503/TR(NTOD)
      COMMON/C0504/TZ(NTOD)
      COMMON/C0505/TC(NTOD)
      COMMON/C0506/ER(NTOD)
      COMMON/C0507/FIRB(NTOD) 
      COMMON/C0508/FIZB(NTOD) 
      COMMON/C0509/FIRG(NTOD) 
      COMMON/C0510/FIZG(NTOD) 
      COMMON/C0512/ENM1(NTOD) 
      COMMON/C0513/ESECN(NTOD)
      COMMON/C0601/ERN(NTOD)
      COMMON/C0602/FIRBN(NTOD)
      COMMON/C0603/FIZBN(NTOD)
      COMMON/C0604/FIRGN(NTOD)
      COMMON/C0605/FIZGN(NTOD)
      COMMON/C0608/RN(NTOD)
      COMMON/C0609/ZN(NTOD)
      COMMON/C0610/RON(NTOD)
      COMMON/C0611/TN(NTOD)
      COMMON/C0612/E(NTOD)
      COMMON/C0613/EN(NTOD)
      COMMON/C0614/PN(NTOD)
      COMMON/C0615/P(NTOD)
      COMMON/C0616/DELTAE(NTOD)
      COMMON/C0801/COE1(NTOD) 
      COMMON/C0802/COE2(NTOD) 
      COMMON/C0803/COE3(NTOD) 
      COMMON/C0804/COE4(NTOD) 
      COMMON/C0805/CB(NTOD)
      COMMON/C0806/CG(NTOD)
      COMMON/C0807/CC(NTOD)
      COMMON/C0808/CD(NTOD)
      COMMON/C0809/CH(NTOD)
      COMMON/C0810/CM(NTOD)
      COMMON/C0811/XDT(NTOD)
      COMMON/C0812/D1(NTOD)
      COMMON/C0813/D2(NTOD)
      COMMON/C0814/D3(NTOD)
      COMMON/C0815/D4(NTOD)
      COMMON/C0818/SNE(NTOD)
      COMMON/CONTI/ 
C  MONITEUR DU STEP3
     *IZZZZZ,IXXXXX,IYYYYYO,NRET,IRET,TITER,NINTOT,INVM
     *,IAPO,NOPROT,AMAXIT,ATPSMAX,ATBMAX,ADTMIN,ACLE1,ATLIM,IZZZZA
     *,TIYYYY,TIXXX,TIZZZZ,TIACTU,TISOR 
      LOGICAL AMAXIT,ATPSMAX,ATBMAX,ADTMIN,ACLE1,ATLIM,IZZZZA
      LOGICAL IZZZZZ,IXXXXX,IYYYYYO,IAPO,NOPROT
      COMMON/ZZZZ/ALK,OMK,APS,ALOM,P75,APH,CPH,ACPH,NINT
     *          ,ISYMA,NKMIN,NLMIN,NKPREM,NLPREM
      COMMON/MILIEU/
     *             NOMMAT(50),METAT(50),MOPAC(50),MDSUP(50),
     *             ROO(50),TOO(50),
     *                  PO(50),
     *             INDOPA(50),INDNEU(50),INDXXX(50),
     *              NCOMP(5,50),TAUX(5,50),
     *             NMPOL(50),NETMEL(5,2),NETPOL(5,2),IDGROU(2),ALFGR, 
     *             FLUB(50),FLUG(50),FLUD(50),FLUH(50),
     .NPARM (50),ICORPS (8,50),YNI (8,50),
     .NPARMP(50),ICORPSP(8,50),YNIP(8,50),
     *AMASSE(50),IREA(8,50),NCOMPP(5,5),TAUXP(5,5),
     *NOFUS(50),IANI(50),
     *ENO(50),
     *MPOLOPA(5),XNACM(50),
     * MILCOD(50),
     *             DERMIL
      COMMON/THERM/ 
     *             NDEB,NFIN,KONV,NOCONV,TETA,JOB,EPS,TESTM,ITERQ
      DIMENSION XTEMP(1)
      EQUIVALENCE (XTEMP(1),FIZG(1))
C
      npasmax=10
      print *,' benchmark a npasmax= ',npasmax
C      DO 1111 NPAS=1,npasmax
C
      NCOL=NNCOL
      NLIG=NNLIG
      NMAT1=1
      NMAT2=6
      NMAT3=11
cbench      NOMMAT(1)=2HL1
cbench      NOMMAT(2)=2HL2
cbench      NOMMAT(3)=2HL3
      NTOTM=NTOD
      NTOT=NCOL*NLIG
      YYYY=.FALSE.
      MOVE=.FALSE.
      ZZZZ=.TRUE.
      T2003=0.
      TEMPS=0.
      TIMEF=0.
      TIMEH=0.
      TIMEN=0.
      DTFS=0.1E-9
      DTHS=0.1e-09
      DTNS=0.1e-09
      DTF=DTFS
      NINTO=0
      INVTOT=0
      NITER=0
      NITERF=0
      MAXITER=300
      TARRET=100.E-9
      MZZZZZ=1
      DTMIN=0.01E-9 
      DTMAXF=2.5E-9 
      TETA=0.5
      MINVER=1
      ISYM=1
      IMULTI=0
      IAXE=1
      P75=7.5
      CPH=3.E+10
      APH=7.56E-15
      ACPH=3./(4.*APH*CPH)
      MNC=NTOT
      DO 1 J=1,NLIG 
      JCOL=(J-1)*NCOL
      DO 1 II=1,NCOL 
      NN=JCOL+II
      KN(NN)=II
      LN(NN)=J
    1 CONTINUE
      DO 2 N=1,NTOT 
    2 MAT(N)=0
      DO 3 N=1,NTOT 
      IF(LN(N).LT.NLIG.AND.KN(N).LT.NCOL)MAT(N)=3 
      IF(LN(N).LT.NLIG.AND.KN(N).LT.NMAT3)MAT(N)=2
      IF(LN(N).LT.NLIG.AND.KN(N).LT.NMAT2)MAT(N)=1
    3 CONTINUE
      DO 4 N=1,NTOT 
      ND(N)=N+1
      IF(KN(N).EQ.NCOL.AND.LN(N).EQ.NLIG)ND(N)=1
      NG(N)=N-1
      IF(KN(N).EQ.1)NG(N)=1
      NB(N)=N-NCOL
      IF(LN(N).EQ.1)NB(N)=1
      NH(N)=N+NCOL
      IF(LN(N).EQ.NLIG)NH(N)=N-NCOL*(NLIG-1)+1
      IF(LN(N).EQ.NLIG.AND.KN(N).EQ.NCOL)NH(N)=1
    4 CONTINUE
      DO 5 N=1,NTOT 
      KLFB(N)=0
      KLFG(N)=0
    5 CONTINUE
      N1=0
      N2=0
      DO 6 N=1,NTOT 
      IF(KN(N).EQ.1.OR.KN(N).EQ.NCOL.OR.LN(N).EQ.NLIG)THEN
      N1=N1+1
      KLFG(N1)=1
      NFG(N1)=N
      ENDIF
      IF(LN(N).EQ.1.OR.LN(N).EQ.NLIG.OR.KN(N).EQ.NCOL)THEN
      N2=N2+1
      KLFB(N2)=1
      NFB(N2)=N
      ENDIF
    6 CONTINUE
      NPFG=N1
      NPFB=N2
      DO 7 N=2,NTOT 
      IVID(N)=15
    7 CONTINUE
      DO 8 N=2,NCOL 
      IVID(N)=12
    8 CONTINUE
      DO 9 N=1,NTOT 
      IF(KN(N).EQ.1)IVID(N)=10
      IF(KN(N).EQ.NCOL)IVID(N)=5
      IF(LN(N).EQ.NLIG)IVID(N)=3
    9 CONTINUE
      IVID(1)=8
      IVID(NCOL)=4
      IVID(NTOT)=1
      IVID(NTOT-NCOL+1)=2
C
      DQEXT=0.
C
      DO 15 N=1,NTOTM
      R(N)=0.
      Z(N)=0.
      RO(N)=1.
      ZMM(N)=1.
      VOLU(N)=1.
      T(N)=100.
      ER(N)=0.
      FIRB(N)=0.
      FIRG(N)=0.
      E(N)=0.
      P(N)=0.
      CB(N)=0.
      CG(N)=0.
      CC(N)=0.
      CD(N)=0.
      CH(N)=0.
      CM(N)=0.
      XDT(N)=0.
      D1(N)=0.
      D2(N)=0.
      D3(N)=0.
      D4(N)=0.
      COE1(N)=1.
      COE2(N)=1.
      COE3(N)=1.
      COE4(N)=1.
      XTEMP(N)=1
      DELTAE(N)=0.
      DWA(N)=0.
      ENM1(N)=0.
      ESECN(N)=0.
      V1(N)=0.
      V2(N)=0.
      V3(N)=0.
      V4(N)=0.
      V5(N)=0.
      V6(N)=0.
      V7(N)=0.
   15 CONTINUE
      DR=NLIG-1
      DR=10./DR
      DZ1=NMAT3-1
      DZ1=10./DZ1
      DZ2=NCOL-NMAT3
      DZ2=0.6/DZ2
      DO 10 N=1,NTOT
      R(N)=(LN(N)-1)*DR
   10 CONTINUE
      Z1=10.
      DO 11 N=1,NTOT
      IF(KN(N).LE.NMAT3) THEN 
      Z(N)=(KN(N)-1)*DZ1
      ELSE
      Z(N)=Z1+(KN(N)-NMAT3)*DZ2
      ENDIF
   11 CONTINUE
      RL=0.7
      RLO=18.9
      DO 12 N=1,NTOT
      IF(KN(N).LT.NMAT3) THEN 
      RO(N)=RL
      ELSE
      RO(N)=RLO
      ENDIF
   12 CONTINUE
      DO 13 N=1,NTOT
      IF(KN(N).LT.NMAT2) THEN 
      T(N)=10.E+6
      ELSE
      T(N)=1.E+6
      ENDIF
   13 CONTINUE
      DO 14 N=1,NTOT
      IF(MAT(N).NE.0) THEN
      R1=R(N)
      Z1=Z(N)
      R2=R(NH(N))
      Z2=Z(NH(N))
      R3=R(NH(ND(N)))
      Z3=Z(NH(ND(N)))
      R4=R(ND(N))
      Z4=Z(ND(N))
      RG1=(R1+R2+R4)/3.
      RG2=(R3+R2+R2)/3.
      S1=((R2-R1)*(Z4-Z1)-(R4-R1)*(Z2-Z1))
      S2=((R4-R3)*(Z2-Z3)-(R2-R3)*(Z4-Z3))
      VOL=(S1*RG1+S2*RG2)*3.14159265
      ZMM(N)=VOL*RO(N)
      VOLU(N)=VOL
      ENDIF
   14 CONTINUE
      TMAX=20.1E+6
      TMIN=600000.
      DTEMP=(TMAX-TMIN)/19
      XT(1)=TMIN
      DO 25 N=2,20
      XT(N)=XT(N-1)+DTEMP
   25 CONTINUE
      ROMIN=0.6
      ROMAX=0.79
      DRO=(ROMAX-ROMIN)/19
      XRO(1)=ROMIN
      DO 26 N=2,20
      XRO(N)=XRO(N-1)+DRO
   26 CONTINUE
      ROMIN=18.8
      ROMAX=18.99
      DRO=(ROMAX-ROMIN)/19
      XRO(21)=ROMIN 
      DO 27 N=22,40 
      XRO(N)=XRO(N-1)+DRO
   27 CONTINUE
      DO 28 I=1,20
      M=1 
      DO 29 J=1,20
      XOPA(I,J)=OPAQ(M,XT(I),XRO(J))
   29 CONTINUE
      M=3 
      DO 30 J=21,40 
      XOPA(I,J)=OPAQ(M,XT(I),XRO(J))
   30 CONTINUE
   28 CONTINUE
      DO 31 N=1,20
      XT(N)=ALOG(XT(N))
   31 CONTINUE
      DO 32 N=1,20
      XRO(N)=ALOG(XRO(N))
   32 CONTINUE
      DO 33 N=21,40 
      XRO(N)=ALOG(XRO(N)+10000.)
   33 CONTINUE
      DO 37 I=1,20
      DO 37 J=1,40
      XOPA(I,J)=ALOG(XOPA(I,J))
   37 CONTINUE
      DO 34 N=1,NTOT
      IF(MAT(N).EQ.3) THEN
      RO(N)=ALOG(RO(N)+10000.)
      ELSE
      RO(N)=ALOG(RO(N))
      ENDIF
   34 CONTINUE
      NBLIG=NLIG-1
      NBKOL=NCOL-1
      DO 20 N=1,NLIG
      NELIG(N)=NBKOL
   20 CONTINUE
      IDLIG(1)=1
      DO 21 N=2,NLIG
      IDLIG(N)=IDLIG(N-1)+NCOL
   21 CONTINUE
      DO 22 N=1,NBKOL
      NECOL(N)=NBLIG
   22 CONTINUE
      I=1 
      DO 23 N=1,NBKOL
      NPREM=N
      DO 24 II=1,NECOL(N)
      KOL(I)=NPREM+(II-1)*NCOL
      I=I+1
   24 CONTINUE
   23 CONTINUE
      ERADO=0.
      EINTO=0.
      ETOTO=0.
      EINT=0.
      ERAD=0.
      ECIN=0.
      DO 16 N=1,NTOT
      IF(MAT(N).NE.0) THEN
      EINTO=EINTO+ZMM(N)*E(N) 
      ERADO=ERADO+ER(N)*VOLU(N)
      ENDIF
   16 CONTINUE
      ETOTO=EINTO+ERADO
C 1111 CONTINUE
C.....REWIND(9).......A NE PAS METTRE SI TAPE9=OUTPUT
      STOP
      END 
      FUNCTION  OPAQ(M,TT,RHO) 
c
c
      PARAMETER (NNLIG=11,NNCOL=31,NTOD=(NNLIG+1)*NNCOL)
       COMMON/MONITOR/
     .LKOM(6),
     1IGENED,
     1 NUPROT,NOMCAS(8),T2003,NINTO,INVTOT,ARRET,MODIF(10),JDATV,NVERS,
     1 NUPRB13,NOMJOB,PGNOM,
     1 DATESG,NOMSAG,NUPRB11,IDATE,ICYCLE,
     1IGENEF,
     2IDIMD,
     2 NMIL,NTOT,NTOTM,NMF,NMAXNL,NFUS,NNTOT,NBNLOG,NBCOL,LIGDER,TRMG,
     2 TRMG2,NBGR,NCORPS,NREA,NSECNE,NNPOL,NNPOLM,NFUSM,
     2IDIMF
      COMMON/MONITOR/
     3LOGIQD,
     3 REPRISE,YYYY,MOVE,ZZZZ,MYYYYYO,MXXXXX,MZZZZZ,IOPACI,IAXE,ISCHEMA,
     3 ISYM,MINVER,ICOURB,SAGACE,IPRLCM,LOI4,LOI7,B13,
     3 VIDEC,COMBLE,PREC,INTNEU,LISS,DECOUPL,SORTZON,NEUDISK,OPADISK, 
     3 B10,DENEU,FUINE,PROTDIS,REPDIS,SANPROT,
     3SORMU,FUS,FUSD,LREMAIL,POLUT,LPRES,LENER,LABAN,
     3 B11,
     3 IFB10,IFRO10,IFEN10,
     3 SANSAG,
     3 METMCC,VARMAIL,
     3IANISO,
     3 IFPART,
     3 LLIBRE(5),LOGIQF
      COMMON /MONITOR/
     4IGESTD,
     4 LARGE(20),NBMOT(20),IADLCM,LCMLON,LON402,
     4L03,L04,L0402,L07,L0822,LETAT,LOPA,LSECT,LVARHE,LTOT,JOPAC,JETAT,
     4JSECT,LXXXPHO,IFLU,ICOU,IFUIT,IYYYY1,INTEG,INTLI,ISORTI,
     4LONFLU,LONCOU,LONFUIT,LONET,LSORTI,NEUDIS,LYYYY,
     4LONMC1,LONMC2,JTIMES,LTIMES,JBILAN,LBILAN,JTDEC,JVRDEC,JVZDEC,
     4 JRDEC,JZDEC,JZONES,JWS1DEC,JWS2DEC,JZMPDEC,
     4 IGESTF
      COMMON /MONITOR/
     5LCOM1,LCOM2,LCOM3,LCOM4,LCOM5,LCOM6,LCOM7,LCOM8,LCOM9,LCOM10,
     .LCREMAY,
     5LCOMFIN
      COMMON/DESCR/KN(NTOD)
     .,LN(NTOD),
     . MAT(NTOD),
     . ND(NTOD),
     . NH(NTOD),
     . NG(NTOD),
     . NB(NTOD),
     . KLFB(NTOD),
     . KLFG(NTOD),
     . IVID(NTOD),
     . DEN(NTOD),MNC,XT(20),XRO(40),XOPA(20,40),OPAK(NTOD), 
     . DP(NTOD),NPFG,NFG(NTOD),NPFB,NFB(NTOD),
     . V1(NTOD),V2(NTOD),V3(NTOD),V4(NTOD),V5(NTOD),V6(NTOD),V7(NTOD),
     . VOLU(NTOD),VOLUN(NTOD),DVOLU(NTOD),
     . NBLIG,NBKOL,NELIG(NTOD),NECOL(NTOD),IDLIG(NTOD),KOL(NTOD)
      COMMON/LICOL/NLIG,NCOL
      INTEGER DATESG
      LOGICAL ARRET 
       LOGICAL REPRISE,YYYY,MOVE,ZZZZ,SAGACE,LOI4,LOI7
      LOGICAL B13,SORTZON,NEUDISK,OPADISK
      LOGICAL B10,DENEU,FUINE,PROTDIS,REPDIS,SANPROT
      LOGICAL IFB10,IFRO10,IFEN10
      LOGICAL SANSAG
      LOGICAL VIDEC,COMBLE,PREC,INTNEU,LISS,DECOUPL,IPRLCM
      LOGICAL SORMU,FUS,FUSD,LREMAIL,POLUT,LPRES,LENER,LABAN
      LOGICAL IANISO
      LOGICAL B11,VARMAIL
      LOGICAL IFPART
C      COMMON/OPA/NROP,IPRO,TDEG,DENS,XLT,NRA,ROA,TA,COMULT
      COMMON/C0101/ISM    (   1)
      COMMON/C0102/NENV   (   1)
      COMMON/C0103/MVKLIM (   1)
      COMMON/C0104/MEKIN  (   1)
      COMMON/C0105/NMETO  (   1)
      COMMON/C0201/R(NTOD)
      COMMON/C0202/Z(NTOD)
      COMMON/C0203/RO(NTOD)
      COMMON/C0204/T(NTOD)
      COMMON/C0205/ZMM(NTOD)
      COMMON/C0206/DWA(NTOD)
C
      COMMON/C0301/XNI    (16,1)
      COMMON/C0303/DNN    (8, 1)
      COMMON/C0901/XNP(16,1)
      COMMON/C1001/DNNFU(5,1) 
      COMMON/C0401/FLUENER(   1)
      COMMON/C0501/WCOUBN (   1)
      COMMON/C0502/WCOUGN (   1)
      COMMON/C0503/TR(NTOD)
      COMMON/C0504/TZ(NTOD)
      COMMON/C0505/TC(NTOD)
      COMMON/C0506/ER(NTOD)
      COMMON/C0507/FIRB(NTOD)
      COMMON/C0508/FIZB(NTOD)
      COMMON/C0509/FIRG(NTOD)
      COMMON/C0510/FIZG(NTOD)
      COMMON/C0511/TTTT   (   1)
      COMMON/C0514/TSG    (   1)
      COMMON/C0515/TSB    (   1)
      COMMON/C0601/ERN(NTOD)
      COMMON/C0602/FIRBN(NTOD)
      COMMON/C0603/FIZBN(NTOD)
      COMMON/C0604/FIRGN(NTOD)
      COMMON/C0605/FIZGN(NTOD)
      COMMON/C0606/VR     (   1)
      COMMON/C0607/VZ     (   1)
      COMMON/C0608/RN(NTOD)
      COMMON/C0609/ZN(NTOD)
      COMMON/C0610/RON(NTOD)
      COMMON/C0611/TN(NTOD)
      COMMON/C0612/E(NTOD)
      COMMON/C0613/EN(NTOD)
      COMMON/C0614/PN(NTOD)
      COMMON/C0615/P(NTOD)
      COMMON/C0616/DELTAE(NTOD)
      COMMON/C0617/DWN    (   1)
      COMMON/C0618/ZMI    (   1)
      COMMON/C1701/WS1(1)
      COMMON/C1702/WS2(1)
      COMMON/C1703/ZMMP1(1)
      COMMON/C1704/WS1N( 1)
      COMMON/C1705/WS2N( 1)
      COMMON/C0701/PRESVIT(   1)
      COMMON/ZZZZ/ALK,OMK,APS,ALOM,P75,APH,CPH,ACPH,NINT
     *          ,ISYMA,NKMIN,NLMIN,NKPREM,NLPREM
      COMMON/TABCOR/
     .IZ(99),XMASION(99),NOM(99),NDEBU(99)
     .,DERCOR
      COMMON/PARTI/XNE,XNIZI2 
      COMMON/MILIEU/
     *             NOMMAT(50),METAT(50),MOPAC(50),MDSUP(50),
     *             ROO(50),TOO(50),
     *                  PO(50),
     *             INDOPA(50),INDNEU(50),INDXXX(50),
     *              NCOMP(5,50),TAUX(5,50),
     *             NMPOL(50),NETMEL(5,2),NETPOL(5,2),IDGROU(2),ALFGR, 
     *             FLUB(50),FLUG(50),FLUD(50),FLUH(50),
     .NPARM (50),ICORPS (8,50),YNI (8,50),
     .NPARMP(50),ICORPSP(8,50),YNIP(8,50),
     *AMASSE(50),IREA(8,50),NCOMPP(5,5),TAUXP(5,5),
     *NOFUS(50),IANI(50),
     *ENO(50),
     *MPOLOPA(5),XNACM(50),
     * MILCOD(50),
     *             DERMIL
C      COMMON/BELLOT/H,EX,C
      COMMON/C0816/PMU    (   1)
      COMMON/C0817/RMU    (   1)
      COMMON/C0818/SNE(NTOD)
      COMMON/C0819/SNT    (   1)
      COMMON/C0820/ATS    (   1)
      COMMON/C0821/ATE    (   1)
      DATA MI/0/
      IF(M.NE.3) THEN
      TX=TT*8.620689E-8
      ALT=ALOG(TX)
      ALR=ALOG(RHO) 
      OPKF=0.229+(0.05*EXP(-ALT*2.9)+6.E-4*EXP(-ALT*5.2)
     .)*EXP(ALR*0.8)
      OPAQ=OPKF*RHO 
      ELSE
      C=0.254E-20
      H=-1.8
      EX=2.24
      XLT=C*RHO**H*TT**EX*3.2808398E+3
      OPAQ=1./XLT
      ENDIF
  100 CONTINUE
      RETURN
      END 