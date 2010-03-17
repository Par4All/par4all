#define TT 1.0594630943592952645618252D0
#define OO 1.1
#define RR 1.234567E0
#define DD 1.234567890123456D0
#define II -1234
#define CC (1.234567E0,-1.234567E0)
#define DC (-1.234567890123456D0,-1.234567890123456D0)
!
! test various constant simplifications
!
      program constants

      integer i
      real r
      complex c
      double precision d
      double complex dc

      i = OO
      r = OO
      d = OO
      c = OO
      dc = OO

      i = TT
      r = TT
      d = TT
      c = TT
      dc = TT

      print *, 'int: ', i
      print *, 'real: ', r
      print *, 'cmplx: ', c
      print *, 'dbl: ', d
      print *, 'dcmplx: ', dc

! double complex promotion?
! it does not seem to work cleanly...
      d = 1.23456789012345D0
      c = (1.23456E0,1.23456E0)
      print *, 'double complex', d + c

! other simplifications

      print *, INT(II)
      print *, REAL(II)
      print *, DBLE(II)
      print *, CMPLX(II)
      print *, DCMPLX(II)

! ambiguous (real*4)?
      print *, INT(OO)
      print *, REAL(OO)
      print *, DBLE(OO)
      print *, CMPLX(OO)
      print *, DCMPLX(OO)

      print *, INT(RR)
      print *, REAL(RR)
      print *, DBLE(RR)
      print *, CMPLX(RR)
      print *, DCMPLX(RR)

      print *, INT(DD)
      print *, REAL(DD)
      print *, DBLE(DD)
      print *, CMPLX(DD)
      print *, DCMPLX(DD)

! too many figures?
      print *, INT(TT)
      print *, REAL(TT)
      print *, DBLE(TT)
      print *, CMPLX(TT)
      print *, DCMPLX(TT)

      print *, INT(CC)
      print *, REAL(CC)
      print *, DBLE(CC)
      print *, CMPLX(CC)
      print *, DCMPLX(CC)

! this extension should be accepted?
      print *, INT(DC)
      print *, REAL(DC)
      print *, DBLE(DC)
      print *, CMPLX(DC)
      print *, DCMPLX(DC)

      end
