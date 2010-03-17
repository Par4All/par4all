
      program x

! 
! tenseur_inc.h (version 1.4)
! 97/01/04, 12:09:00
!
      IMPLICIT NONE
      integer           SIZE_NS,SIZE_NH
      PARAMETER ( SIZE_NS = 64)
      PARAMETER ( SIZE_NH = 32)
      INTEGER nxsall,nysall,nzsall
      REAL*8 pi, rho0, mel, qel, rs
      INTEGER nhfx, nhfy, nhfz, de, nhf, nmax
      INTEGER licn, lirn, nzmax, nbints
      INTEGER ntxt, ntyt, ntzt, nxs, nys, nzs
      INTEGER nbtdmax
      INTEGER npartmax, pcolmax, lanag
      INTEGER nrmax, npmax
      INTEGER nbcbmax
      PARAMETER (nrmax=100, npmax=100)
      PARAMETER (nbtdmax=10000)
      PARAMETER (nhfx=SIZE_NH, nhfy=SIZE_NH, nhfz=SIZE_NH)
      PARAMETER (pi=3.1415926535890)
      PARAMETER (mel=1.0, qel=-1.0)
      PARAMETER (rs=4.0)
      PARAMETER (npartmax=500000)
      PARAMETER (de=2)
      PARAMETER (pcolmax=64)
      PARAMETER (rho0=3.73019397872-03)
      PARAMETER (ntxt=de * nhfx + 1)
      PARAMETER (ntyt=de * nhfy + 1, ntzt=de * nhfz + 1)
      PARAMETER (nxs=de*nhfx, nys=de*nhfy, nzs=de * nhfz )
      PARAMETER (nhf=nhfx)
      PARAMETER (nbints=2000)
      PARAMETER (nbcbmax=10000)

      INTEGER NS
      PARAMETER (NS=64)


!hpf$ processors PT3D(7)
!!!hpf$ processors PT3D(15)


!hpf$ template T(nxs,nys,nzs)

!
! 4 processors
!
!!!hpf$ processors P(4)
!hpf$ processors P(2,2)

!
! 8 processors
!
!!!hpf$ processors P(8)
!!!hpf$ processors P(4,2)
!!!hpf$ processors P(2,2,2)

!
! template distribution
!
!!!hpf$ distribute T(*,*,block) onto P
!hpf$ distribute T(*,block,block) onto P
!!!hpf$ distribute T(block,block,block) onto P

!
! end of tenseur_inc.h
!

      REAL*8 mx(nxs, nxs)
      REAL*8 mxt(nxs, nxs)
      INTEGER i, j, k

!hpf$ align mx(*,i), mxt(i,*) with T(i,*,*)

!hpf$ independent
      DO j = 1, nsx
!hpf$    independent
         DO i = 1, nsx
            mx(i, j) = float(i + j)
         ENDDO
      ENDDO
      CALL transpose(nsx, mx, mxt)
      END

      SUBROUTINE transpose(nsx, matx, matxt)

      integer nsx, i, j

! 
! tenseur_inc.h (version 1.4)
! 97/01/04, 12:09:00
!
      IMPLICIT NONE
      integer           SIZE_NS,SIZE_NH
      PARAMETER ( SIZE_NS = 64)
      PARAMETER ( SIZE_NH = 32)
      INTEGER nxsall,nysall,nzsall
      REAL*8 pi, rho0, mel, qel, rs
      INTEGER nhfx, nhfy, nhfz, de, nhf, nmax
      INTEGER licn, lirn, nzmax, nbints
      INTEGER ntxt, ntyt, ntzt, nxs, nys, nzs
      INTEGER nbtdmax
      INTEGER npartmax, pcolmax, lanag
      INTEGER nrmax, npmax
      INTEGER nbcbmax
      PARAMETER (nrmax=100, npmax=100)
      PARAMETER (nbtdmax=10000)
      PARAMETER (nhfx=SIZE_NH, nhfy=SIZE_NH, nhfz=SIZE_NH)
      PARAMETER (pi=3.1415926535890)
      PARAMETER (mel=1.0, qel=-1.0)
      PARAMETER (rs=4.0)
      PARAMETER (npartmax=500000)
      PARAMETER (de=2)
      PARAMETER (pcolmax=64)
      PARAMETER (rho0=3.73019397872-03)
      PARAMETER (ntxt=de * nhfx + 1)
      PARAMETER (ntyt=de * nhfy + 1, ntzt=de * nhfz + 1)
      PARAMETER (nxs=de*nhfx, nys=de*nhfy, nzs=de * nhfz )
      PARAMETER (nhf=nhfx)
      PARAMETER (nbints=2000)
      PARAMETER (nbcbmax=10000)

      INTEGER NS
      PARAMETER (NS=64)


!hpf$ processors PT3D(7)
!!!hpf$ processors PT3D(15)


!hpf$ template T(nxs,nys,nzs)

!
! 4 processors
!
!!!hpf$ processors P(4)
!hpf$ processors P(2,2)

!
! 8 processors
!
!!!hpf$ processors P(8)
!!!hpf$ processors P(4,2)
!!!hpf$ processors P(2,2,2)

!
! template distribution
!
!!!hpf$ distribute T(*,*,block) onto P
!hpf$ distribute T(*,block,block) onto P
!!!hpf$ distribute T(block,block,block) onto P

!
! end of tenseur_inc.h
!

      REAL*8 matx(nxs, nxs), matxt(nxs, nxs)

!hpf$ align matx(*,i), matxt(i,*) with T(i,*,*)

!hpf$ independent
      DO i = 1, NS
!hpf$    independent
         DO j = 1, NS
            matxt(j, i) = matx(i, j)
         ENDDO
      ENDDO

      END
