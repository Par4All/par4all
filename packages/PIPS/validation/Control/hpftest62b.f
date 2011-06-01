c unstructured
      program hpftest62b
      integer n,m
      parameter (n=10)
      print *, 'hpftest62 running'
      if (.true.) then
 10     print *, 'please enter a value'
        read *, m
        if (m.gt.n.or.m.lt.1) goto 10
      endif
      print *, 'ok : ', m
      end
