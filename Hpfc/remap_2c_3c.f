      program cyclic_remap
      integer n
      parameter (n=60)
      real a(n)
chpf$ template t(n)
chpf$ dynamic t, a
chpf$ align a(i) with t(i)
chpf$ processors p2(2)
chpf$ processors p3(3)
chpf$ distribute t(cyclic) onto p2
      print *, a(22)
chpf$ redistribute t(cyclic) onto p3
      print *, a(33)
      end
