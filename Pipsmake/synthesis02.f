      program synthesis02

C     Goal: debug the code synthesis in initializer()
C     Check that a call graph of this routine can be obtained
C     by automatically generating source code for FOO and BAR

      complex c

      character*80 s
c      parameter (lng=10)
C      character*lng s2

      call foo(x,i,c,s)

c      c = bar(i, s2)
      c = bar(i,j,k)

      end
