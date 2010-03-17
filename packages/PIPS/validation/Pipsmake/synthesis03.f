      program synthesis03

C     Goal: debug the code synthesis in initializer()
C     Check that a call graph of this routine can be obtained
C     by automatically generating source code for FOO and BAR

C     Handle arrays as actual arguments and actual expressions
c     whose type should be derived

      complex c(10)

      character*80 s

      call foo(x,i,c,s)

      c = bar(i+1)

      end
