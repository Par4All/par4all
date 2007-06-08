      program ca
      parameter (mn=10)
      dimension a(mn)
      im = mn
      nmax = 6
      do 100 i = 1, nmax
         call sub(im,a)
 100  continue
      do 101 i = 1,im
         print *,a(i)
 101  continue
      end
c
      subroutine sub(imm,b)
      dimension b(imm)
      mk = imm -1
      mm = 2*mk
      do 200 i = 1, mm
         b(i) = b(i) + 2.
 200  continue
      return
      end
