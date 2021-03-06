      SUBROUTINE OAVITEL(IFM,JFM)
C
      PARAMETER(NFAC=49)
      COMMON/OO/O(NFAC),VS(NFAC),ps(nfac)
      COMMON/PCP/XO(NFAC),YO(NFAC),ZO(NFAC)
      COMMON/GRA/GRD(NFAC,3)
      COMMON/CFN/FNX(NFAC),FNY(NFAC),FNZ(NFAC),AIRE(NFAC)
C
C
      IMFM1=IFM-1
      JMFM1=JFM-1
      JM1O2=JMFM1/2
      JM1O2M1=JM1O2-1
      JM1O2P3=JM1O2+3
C
      IFAC=1
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=XO(IFAC+1)-XO(IFAC)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=YO(IFAC+1)+YO(IFAC)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=ZO(IFAC+1)-ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC))
      DM2=O(IFAC+1)-O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 19 J=2,JM1O2M1
      IFAC=IFAC+1
C
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=XO(IFAC+1)-XO(IFAC-1)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=YO(IFAC+1)-YO(IFAC-1)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=ZO(IFAC+1)-ZO(IFAC-1)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC)
      DM2=O(IFAC+1)-O(IFAC-1)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   19 CONTINUE
C
      IFAC=IFAC+1
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=-4.*XO(IFAC-1)+XO(IFAC-2)+3.*XO(IFAC)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=-4.*YO(IFAC-1)+YO(IFAC-2)+3.*YO(IFAC)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=-4.*ZO(IFAC-1)+ZO(IFAC-2)+3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC)
      DM2=-4.*O(IFAC-1)+O(IFAC-2)+3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      IFAC=IFAC+1
C
      IFAC=IFAC+1
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=4.*XO(IFAC+1)-XO(IFAC+2)-3.*XO(IFAC)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=4.*YO(IFAC+1)-YO(IFAC+2)-3.*YO(IFAC)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=4.*ZO(IFAC+1)-ZO(IFAC+2)-3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC))
      DM2=4.*O(IFAC+1)-O(IFAC+2)-3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 18 J=JM1O2P3,JMFM1
      IFAC=IFAC+1
C
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=XO(IFAC+1)-XO(IFAC-1)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=YO(IFAC+1)-YO(IFAC-1)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=ZO(IFAC+1)-ZO(IFAC-1)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC)
      DM2=O(IFAC+1)-O(IFAC-1)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   18 CONTINUE
C
      IFAC=IFAC+1
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=-XO(IFAC-1)+XO(IFAC)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=-YO(IFAC-1)-YO(IFAC)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=-ZO(IFAC-1)+ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC)
      DM2=-O(IFAC-1)+O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 39 I=2,IMFM1
      IFAC=IFAC+1
C
      R1=(XO(IFAC+JFM)-XO(IFAC-JFM))
      R2=XO(IFAC+1)-XO(IFAC)
      R4=(YO(IFAC+JFM)-YO(IFAC-JFM))
      R5=YO(IFAC+1)+YO(IFAC)
      R7=(ZO(IFAC+JFM)-ZO(IFAC-JFM))
      R8=ZO(IFAC+1)-ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IFAC+JFM)-O(IFAC-JFM))
      DM2=O(IFAC+1)-O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
C POINT COURANT
C
      DO 29 J=2,JM1O2M1
      IFAC=IFAC+1
C
      IP1=IFAC+JFM
      IM1=IFAC-JFM
      JP1=IFAC+1
      JM1=IFAC-1
      R1=(XO(IP1)-XO(IM1))
      R2=(XO(JP1)-XO(JM1))
      R4=(YO(IP1)-YO(IM1))
      R5=(YO(JP1)-YO(JM1))
      R7=(ZO(IP1)-ZO(IM1))
      R8=(ZO(JP1)-ZO(JM1))
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IP1)-O(IM1))
      DM2=(O(JP1)-O(JM1))
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   29 CONTINUE
C
      IFAC=IFAC+1
      R1=(XO(IFAC+JFM)-XO(IFAC-JFM))
      R2=-4.*XO(IFAC-1)+XO(IFAC-2)+3.*XO(IFAC)
      R4=(YO(IFAC+JFM)-YO(IFAC-JFM))
      R5=-4.*YO(IFAC-1)+YO(IFAC-2)+3.*YO(IFAC)
      R7=(ZO(IFAC+JFM)-ZO(IFAC-JFM))
      R8=-4.*ZO(IFAC-1)+ZO(IFAC-2)+3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IFAC+JFM)-O(IFAC-JFM))
      DM2=-4.*O(IFAC-1)+O(IFAC-2)+3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      IFAC=IFAC+1
C
      IFAC=IFAC+1
      R1=(XO(IFAC+JFM)-XO(IFAC-JFM))
      R2=4.*XO(IFAC+1)-XO(IFAC+2)-3.*XO(IFAC)
      R4=(YO(IFAC+JFM)-YO(IFAC-JFM))
      R5=4.*YO(IFAC+1)-XO(IFAC+2)-3.*XO(IFAC)
      R7=(ZO(IFAC+JFM)-ZO(IFAC-JFM))
      R8=4.*ZO(IFAC+1)-ZO(IFAC+2)-3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IFAC+JFM)-O(IFAC-JFM))
      DM2=4.*O(IFAC+1)-O(IFAC+2)-3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 28 J=JM1O2P3,JMFM1
      IFAC=IFAC+1
C
      IP1=IFAC+JFM
      IM1=IFAC-JFM
      JP1=IFAC+1
      JM1=IFAC-1
      R1=(XO(IP1)-XO(IM1))
      R2=(XO(JP1)-XO(JM1))
      R4=(YO(IP1)-YO(IM1))
      R5=(YO(JP1)-YO(JM1))
      R7=(ZO(IP1)-ZO(IM1))
      R8=(ZO(JP1)-ZO(JM1))
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IP1)-O(IM1))
      DM2=(O(JP1)-O(JM1))
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   28 CONTINUE
C
      IFAC=IFAC+1
C
      R1=(XO(IFAC+JFM)-XO(IFAC-JFM))
      R2=-XO(IFAC-1)+XO(IFAC)
      R4=(YO(IFAC+JFM)-YO(IFAC-JFM))
      R5=-YO(IFAC-1)-YO(IFAC)
      R7=(ZO(IFAC+JFM)-ZO(IFAC-JFM))
      R8=-ZO(IFAC-1)+ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IFAC+JFM)-O(IFAC-JFM))
      DM2=-O(IFAC-1)+O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   39 CONTINUE
C
      IFAC=IFAC+1
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=XO(IFAC+1)-XO(IFAC)
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=YO(IFAC+1)+YO(IFAC)
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=ZO(IFAC+1)-ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=O(IFAC+1)-O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 49 J=2,JM1O2M1
      IFAC=IFAC+1
C
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=(XO(IFAC+1)-XO(IFAC-1))
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=(YO(IFAC+1)-YO(IFAC-1))
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=(ZO(IFAC+1)-ZO(IFAC-1))
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=(O(IFAC+1)-O(IFAC-1))
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   49 CONTINUE
C
      IFAC=IFAC+1
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=-4.*XO(IFAC-1)+XO(IFAC-2)+3.*XO(IFAC)
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=-4.*YO(IFAC-1)+YO(IFAC-2)+3.*YO(IFAC)
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=-4.*ZO(IFAC-1)+ZO(IFAC-2)+3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=-4.*O(IFAC-1)+O(IFAC-2)+3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      IFAC=IFAC+1
C
      IFAC=IFAC+1
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=4.*XO(IFAC+1)-XO(IFAC+2)-3.*XO(IFAC)
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=4.*YO(IFAC+1)-YO(IFAC+2)-3.*YO(IFAC)
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=4.*ZO(IFAC+1)-ZO(IFAC+2)-3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=4.*O(IFAC+1)-O(IFAC+2)-3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 48 J=JM1O2P3,JMFM1
      IFAC=IFAC+1
C
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=(XO(IFAC+1)-XO(IFAC-1))
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=(YO(IFAC+1)-YO(IFAC-1))
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=(ZO(IFAC+1)-ZO(IFAC-1))
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=(O(IFAC+1)-O(IFAC-1))
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   48 CONTINUE
C
      IFAC=IFAC+1
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=-XO(IFAC-1)+XO(IFAC)
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=-YO(IFAC-1)-YO(IFAC)
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=-ZO(IFAC-1)+ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=-O(IFAC-1)+O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      RETURN
      END
