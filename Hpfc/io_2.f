c io - there was a bug
      program io_2
      integer n
      parameter (n=20)
      integer A(n)
      integer i, l, u
chpf$ processors p(4)
chpf$ template t(n)
chpf$ align a with t
chpf$ distribute t(block) onto p
chpf$ independent
      do i=1, n
         a(i) = i+0.1
      enddo
      l = (n/2)-3
      u = (n/2)+2
      print *, (a(i), i=l, u)
      end
