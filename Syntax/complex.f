      program complex
C     bug: complex constant are not recognized by PIPS's parser
C          they have to be replaced by a call to intrinsic CMPLX
      complex cmplx2
c     cmplx2 = (0.1,0.)
      cmplx2 = cmplx(0.1,0.)
      end
