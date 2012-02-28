! common which depends on a parameter...
      program decl7

      parameter (n=5)
      common /foo/ i1(n)

      integer p
      parameter (m=1,p=2)
      common /foo2/ j(m),k(p)
      
      integer q, qx
      parameter (q=10, qx=12)
      common /foo3/ l(q)

      print *, i1(1), j(1), l(2)
      call bla
      end

      subroutine bla

      parameter (n=5)
      common /foo/ i1(n)

      integer p
      parameter (m=2,p=1)
      common /foo2/ j(m),k(p)

      integer q, qx
      parameter (q=3, qx=10)
      common /foo3/ l(qx)

      print *, i1(1), k(1), l(2)
      end
