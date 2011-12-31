import pyps
import os
pyps.workspace.delete("basics4")
w=pyps.workspace("basics3.c",name="basics4",deleteOnClose=True)
m=w.fun.main
print m.cu
print m.workspace.name
print m.name
m.edit(editor="sed -i 's/h/b/'")
m.display()
m.run(["sed","s/b/c/"])
m.display()
m.show("PRINTED_FILE")
print str(m.code)
m.display()
m.code="""int main() {
for(int j=0;j<10;j++)
for(int k=0;k>-6;k--)
puts("ah");
}
"""
m.display()
for i in m.loops():print i.label
for i in m.inner_loops():print i.label
map(pyps.modules.display, m.callers)
map(pyps.modules.display, m.callees)

m.saveas(os.path.join(w.tmpdirname,"a.c"))
w.close()
