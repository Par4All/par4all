c unstructured
      program hpftest62
      integer n,m
      parameter (n=10)
      print *, 'hpftest62 running'
 10   print *, 'please enter a value'
      read *, m
      if (m.gt.n.or.m.lt.1) goto 10
      print *, 'ok : ', m
      end
