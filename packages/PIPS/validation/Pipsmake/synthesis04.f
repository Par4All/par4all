      program synthesis04

C     Goal: debug the code synthesis in initializer()
C     Check that a call graph of this routine can be obtained

C     A warning should be generated because foo cannot be properly typed

      call foo(x)

      call foo(i)

      call foo(x, j)

      end
