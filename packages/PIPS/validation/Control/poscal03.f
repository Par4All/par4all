C     Derived from POSCAL, but simplified

      SUBROUTINE POSCAL03 (NFICAL,NAMECA,NAMECB,FINCAL,                           
     *     LNAMEA,LNAMEB,ITABLE,IARRET,IRETOU,NFDIAG)             

      LOGICAL         LABSOL
      COMMON /REDCOM/ FLOTAN,FL0TAN,INTEGR,INTEGS,ITYPLU,ITYPLV,NCARLU
     *     ,NCARLV,IENDFI,ILECSF(99,3),NREPET,NFILEC,IOPERA,NOTRLC
     *     ,IOPANT,NBOUKL,FACTOR,ICARES,LABSOL,NBOUPA

      CHARACTER * 132  TEXTLU,TITRE,LIGNE,TXLIGN,TEXTLV
      COMMON  /REDTEX/ TEXTLU,TITRE,LIGNE,TXLIGN(99),TEXTLV

      LOGICAL         IMPTES, LTESTM                                            
      CHARACTER *  80 FINCAL, NAMECA, NAMECB                                    
      CHARACTER * 132 TITREC                                                    

c     5             if the next line is commented out, the controlizer bug disappears...
C     Same thing if the other READ is commented out
      READ (NFICAL,'(A)',END=9999) TXLIGN (NFICAL)
      DO   WHILE (TXLIGN(NFICAL)(:1).EQ.'(' .OR.                     
     *     TXLIGN(NFICAL)(:1).EQ.'*')                         
         READ (NFICAL,'(A)',END=9999) TXLIGN (NFICAL)              
      ENDDO                                                          
 9999 IRETOU = 1                                                                
      END                                                                       
