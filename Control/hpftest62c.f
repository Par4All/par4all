c unstructured
      program hpftest62c
      integer n,m
      parameter (n=10)
      print *, 'hpftest62 running'
cfcd$ host
 10   print *, 'please enter a value'
      read *, m
      if (m.gt.n.or.m.lt.1) goto 10
cfcd$ end host
      print *, 'ok : ', m
      end
