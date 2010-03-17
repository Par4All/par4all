      program essai
c
      integer i, j, k
c
      do i=1, 1000
      if (i.gt.500) call P(i)
      enddo
c
      call P(10)
c
c
      j=1
      call Q(j)
c
      k=1
      call R(k)
c
      end
c
c
c
      subroutine P(x)
      call R(x)
      x = 1 + S(0,1)
      end
c
      subroutine Q(y)
      call P(y+1)
      end
c
      subroutine R(z)
      do k=1, 100
      z=z*2
      enddo
      call T(z)
      end
c
      function S(a,b)
      S = (a + 3) / b
      end
c
      subroutine T(w)
      w = w*w
      end
