      program type29

C     Check analysis of non affine arguments

      real x, y, u, v
      logical l, l1, l2, l3
      character*4 s, s1, s2, s3

      u = 2.
      v = 3.
      s2 = "Hi!"
      l2 = .TRUE.

      call copy_four(i, 1+mod(j, 2), x, u+v, s1, s2, l1, l2.AND.l3)

      print *, i, x, s1, l1

      read *, j, u, v, s2, l2

      print *, i, x, s1, l1

      end

      subroutine copy_four(inew, iold,
     &     xnew, xold, snew, sold, lnew, lold)

      character*4 snew, sold
      logical lnew, lold

      inew = iold
      xnew = xold
      snew = sold
      lnew = lold

      end

