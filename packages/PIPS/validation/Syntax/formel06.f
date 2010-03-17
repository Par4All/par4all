      PROGRAM formel06

C     Goal: find out that C is a function after the call to foo

      integer un
      parameter (un=1)

      j = un
      call foo(C)
      I = C(un)

      end
