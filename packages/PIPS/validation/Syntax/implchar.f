      program implchar
c      subroutine implchar

C     Extention of PIPS Fortran required by Nicky Preston-Williams
C     Cachan bug 11: handling of character type in implicit

      parameter (m=10)

      implicit character*8 (d)
      implicit complex*8 (z)

c     This is forbidden by the standard:
c      implicit character*m (e)

c      save
      real t(m)
c      common /foo/do, de, t
c
      do = 'Hello'

      de = 'world!'

c     Type clash, detected by g77, not by f77
c      erreur = 'rate!'

      t(1) = 1.

      call dummy(i, j)

      print *, do, de, erreur, t(1), dance(i, j, k)

      end

      subroutine dummy(i, j)
      print *, i+j
      end

      character*8 function dance(i, j, k)
      dance = 'Bye!'
      end
