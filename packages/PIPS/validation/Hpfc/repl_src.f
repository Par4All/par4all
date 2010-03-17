c remapping with replicated source and target
      program replicated_source
      integer n
      parameter(n=20)
      real A(n)
chpf$ template T(n,n,n)
chpf$ dynamic A, T
chpf$ processors Ps(2,2,2)
chpf$ processors Pt(5,2)
chpf$ align A(i) with T(*,i,*)
chpf$ distribute T(block,block,block) onto Ps
      A(5) = 5.0
chpf$ redistribute T(*,cyclic(2),block) onto Pt
      A(5) = A(5) + 3.1
      end
