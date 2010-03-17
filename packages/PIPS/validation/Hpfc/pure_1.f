c test pure directive
      program pure_1
      external cube
      real cube
cfcd$ pure cube
      real A(100)
chpf$ processors P(4)
chpf$ template T(100)
chpf$ align A with T
chpf$ distribute T(block) onto P
      integer i
chpf$ independent
      do i=1, 100
        A(i) = 1.0/real(i)
      enddo
chpf$ independent
      do i=1, 100
        A(i) = cube(A(i))
      enddo
      print *, A(51)
      end
      real function cube(x)
cfcd$ pure
      real x
      cube=x*x*x
      end
