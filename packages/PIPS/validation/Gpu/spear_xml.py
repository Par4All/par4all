from validation import vworkspace
import os

with vworkspace() as w:

  for fun in w.all_functions:
    fun.gpu_xml_dump()
    print file(os.path.join(w.dirname,fun.show("GPU_XML_FILE"))).read()
