C     excerpt from the Perfect Club program BDNA
C     interest: test of the dependence graph with regions
C
      SUBROUTINE PARPOT(NEU,NLU,NT1,NT2)                                
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)                               
      CHARACTER*4 ZITES,TYPE,ESORT,LSORT,NT1,NT2,NEU,NLU                
      DIMENSION ZITES(9),TYPE(3),ESORT(3),LSORT(2)                      
      SAVE ZITES, TYPE, ESORT, LSORT                                    
      DIMENSION CHRG(11)                                                
      LOGICAL        MC,MD,LRES,LSAVE,LSPA,INMC,LTS,NOTR,LGR,LWW        
     X              ,LWC1,LWA1,LWCN,LWAN,LWC1A1,LQ2PC,LCA,LWWW,LLF,LPC  
      COMMON/LCONTR/ MC,MD,LRES,LSAVE,LSPA,INMC,LTS,NOTR,LGR,LWW        
     X              ,LWC1,LWA1,LWCN,LWAN,LWC1A1,LQ2PC,LCA,LWWW,LLF,LPC  
      COMMON/INDEXQ/ NWI,NWJ,NWK,INDEX(11,11),INDS(3,10),NONE           
      COMMON/TPARAM/ NCPBF(3),NPBF,NQ                                   
      COMMON/TABELS/ QQ(11,11),A(5),B(5)                                
      COMMON/PARAMS/ A1,A2,A3,A4,A5,B1                                  
      COMMON/PHYSIC/ TTRAN(3),TROT(3),RHO,VOLM,DT,FNOP,BREAK            
     X              ,DM(3,3),QM(9,3),TE(3),RE(3),TTS(3),RTS(3)          
     X              ,R2(3),EWW,EWI,EWA,EII,EIA                          
      COMMON/CONTRL/NSTEPS,NSTEP,NSBPO,NSBTS,NSSTS,NSTCK,NTYPES,NOP     
     X              ,NSITES(3),NSPECI(3),NSITET,NENT,NATOMS,NION        
     X              ,NDIM,NSS,NAVRS,NOP3,NOP4,NSS3,NPOCS                
      COMMON/CONSTS/ AVSNO,BOLTZ,PLANCK,PI,ELECHG,EPS0                  
      COMMON/CONVRT/ DSUMM(3),DRI(3,3),DQ(30),DSITE(3,30)               
     X              ,UNITM,UNITL,UNITE,COULF,ENERF,TEMPF,TSTEP          
      DATA ZITES/'   O','  H1','  H2','   P','   M','  M1','  M2',      
     X'+ION','-ION'/                                                    
      DATA TYPE /' H2O','+ION','-ION'/                                  
      DATA ESORT /'KCAL','MHAR','HART'/                                 
      DATA LSORT /'ANGS','A.U.'/                                        
      A(1)=1088213.2D0                                                  
      A(2)=1455.427D0                                                   
      A(3)=-273.5954D0                                                  
      A(4)=666.3373D0                                                   
      A(5)=0.D0                                                         
      B(1)=5.152712D0                                                   
      B(2)=2.961895D0                                                   
      B(3)=2.233264D0                                                   
      B(4)=2.760844D0                                                   
      B(5)=0.D0                                                         
      NT1= TYPE(1)                                                      
      NT2=TYPE(1)                                                       
      NPBF=3                                                            
      NCPBF(1)=1                                                        
      NCPBF(2)=2                                                        
      NCPBF(3)=1                                                        
      NEU=ESORT(1)                                                      
      NLU=LSORT(1)                                                      
      CHRG(1)=0.D0                                                      
      CHRG(2)=0.7175D0                                                  
      CHRG(3)=0.7175D0                                                  
      CHRG(4)=-1.435D0                                                  
      DO 100 I=1,11                                                     
      DO 100 J=1,11                                                     
  100 INDEX(I,J)=4                                                      
      INDEX(1,3)=2                                                      
      INDEX(2,2)=3                                                      
      INDEX(2,3)=3                                                      
      INDEX(3,3)=3                                                      
      DO 110 I=1,3                                                      
      DO 110 J=I,3                                                      
  110 INDEX(J,I)=INDEX(I,J)                                             
      DO 120 I=1,4                                                      
      DO 120 J=1,4                                                      
      QQ(I,J)=CHRG(I)*CHRG(J)                                           
  120 CONTINUE                                                          
      M=4                                                               
      WRITE(66,1505)                                                    
      WRITE(66,1510)                                                    
 1505 FORMAT(////20X,' P O T E N T I A L   M O D E L       : '//)       
 1510 FORMAT(/20X,' NON-COULOMB INTERACTION INDEX TABLE: '//)           
      WRITE(66,1520) (I,I=1,M)                                          
 1520 FORMAT(/20X,3X,10I5//)                                            
      DO 1530 I=1,M                                                     
 1530 WRITE(66,1540)I,(INDEX(I,J),J=1,M)                                
 1540 FORMAT(20X,I2,1X,10I5)                                            
      WRITE(66,1550)                                                    
 1550 FORMAT(//20X,' CHARGES ON COULOMB TERMS:'//)                      
      WRITE(66,1560) (ZITES(I),I=1,M)                                   
 1560 FORMAT(20X,10(5X,A4)//)                                           
      QSUM=0.D0                                                         
      DO 1570 I=1,M                                                     
      WRITE(66,1580) ZITES(I),(QQ(I,J),J=1,M)                           
      DO 1570 J=1,M                                                     
      QSUM=QSUM+QQ(I,J)                                                 
 1570 CONTINUE                                                          
 1580 FORMAT(20X,A4,10(1X,F8.5))                                        
      WRITE(66,1585) QSUM                                               
 1585 FORMAT(/20X,'..SUM OVER POINT CHARGES =',F5.2,' (SHOULD BE ZERO)')
      DO 150 I=1,4                                                      
      DO 150 J=1,4                                                      
      QQ(I,J)=QQ(I,J)*COULF                                             
  150 CONTINUE                                                          
      RETURN                                                            
      END     
