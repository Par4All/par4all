! test complex constants
      program cc
      complex c
      double complex dc
! simple complex conversions
      c = 1
      c = 1.0
      c = 1.0e0
      c = 1.0d0
! implicit complex
      c = (1, 1)
      c = (1.0, 1.0)
      c = (1.0e0, 1.0e0)
      c = (1.0d0, 1.0d0)
! mixed complex cases
      c = (1.0, 1)
      c = (1.0e0, 1)
      c = (1.0d0, 1)
      c = (1, 1.0)
      c = (1.0e0, 1.0)
      c = (1.0d0, 1.0)
      c = (1, 1.0e0)
      c = (1.0, 1.0e0)
      c = (1.0d0, 1.0e0)
      c = (1, 1.0d0)
      c = (1.0, 1.0d0)
      c = (1.0e0, 1.0d0)
! explicit complex
      c = cmplx(1, 1)
      c = cmplx(1.0, 1.0)
      c = cmplx(1.0e0, 1.0e0)
      c = cmplx(1.0d0, 1.0d0)
! mixed complex cases
      c = cmplx(1.0, 1)
      c = cmplx(1.0e0, 1)
      c = cmplx(1.0d0, 1)
      c = cmplx(1, 1.0)
      c = cmplx(1.0e0, 1.0)
      c = cmplx(1.0d0, 1.0)
      c = cmplx(1, 1.0e0)
      c = cmplx(1.0, 1.0e0)
      c = cmplx(1.0d0, 1.0e0)
      c = cmplx(1, 1.0d0)
      c = cmplx(1.0, 1.0d0)
      c = cmplx(1.0e0, 1.0d0)
! simple dcomplex conversions
      dc = 1
      dc = 1.0
      dc = 1.0e0
      dc = 1.0d0
! implicit dcomplex
      dc = (1, 1)
      dc = (1.0, 1.0)
      dc = (1.0e0, 1.0e0)
      dc = (1.0d0, 1.0d0)
! mixed dcomplex cases
      dc = (1.0, 1)
      dc = (1.0e0, 1)
      dc = (1.0d0, 1)
      dc = (1, 1.0)
      dc = (1.0e0, 1.0)
      dc = (1.0d0, 1.0)
      dc = (1, 1.0e0)
      dc = (1.0, 1.0e0)
      dc = (1.0d0, 1.0e0)
      dc = (1, 1.0d0)
      dc = (1.0, 1.0d0)
      dc = (1.0e0, 1.0d0)
! explicit dcomplex
      dc = dcmplx(1, 1)
      dc = dcmplx(1.0, 1.0)
      dc = dcmplx(1.0e0, 1.0e0)
      dc = dcmplx(1.0d0, 1.0d0)
! mixed dcomplex cases
      dc = dcmplx(1.0, 1)
      dc = dcmplx(1.0e0, 1)
      dc = dcmplx(1.0d0, 1)
      dc = dcmplx(1, 1.0)
      dc = dcmplx(1.0e0, 1.0)
      dc = dcmplx(1.0d0, 1.0)
      dc = dcmplx(1, 1.0e0)
      dc = dcmplx(1.0, 1.0e0)
      dc = dcmplx(1.0d0, 1.0e0)
      dc = dcmplx(1, 1.0d0)
      dc = dcmplx(1.0, 1.0d0)
      dc = dcmplx(1.0e0, 1.0d0)

      print *, c, dc
      end

