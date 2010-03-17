      program exemplearticle
      integer a, c
      call exemplearticlefoo(a, a, c)
      print*, a, c
      end
      subroutine exemplearticlefoo(a, b, c)
      integer a, b, c
      read *, a
      b = 3 * a
      c = a * b
      end














