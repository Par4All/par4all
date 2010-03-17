      program NOP
      integer n, i
      parameter (n=100)
      real A(n), B(n), C(n), D(n), E(n)
chpf$ processors P(number_of_processors())
chpf$ template T(n)
chpf$ template T2(n)
chpf$ align A with T
chpf$ align B with T2
chpf$ distribute T onto P
chpf$ distribute T2
chpf$ align C with D
chpf$ distribute D
chpf$ distribute E(cyclic)
chpf$ independent
      do i=1, n
        A(i)=1/i
      enddo
      end
