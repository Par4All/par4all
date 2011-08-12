from pyps import workspace, module
import pypsutils
import shutil, os, re
import webbrowser

def ir_navigator(m,openBrowser=False,output_dir="ir_navigator",keep_existing=False,symbol_table=False):
    """Produce IR (internal representation) output for a module"""
    m.html_prettyprint(symbol_table=symbol_table)
    filename = os.path.join(m.workspace.dirname,m.show("HTML_IR_FILE"))
    
    if not os.path.isfile(filename):
        raise RuntimeError("File (" + filename + ") doesn't exists ! Bug inside ?")

    with open(filename, 'r') as f:
        read_data = f.read()
        
    # Create output dir if not existing    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Initialize output directory
    ir_navigator_runtime_dir = pypsutils.get_runtimedir('ir_navigator')
    
    for f in os.listdir(ir_navigator_runtime_dir):
        shutil.copy(os.path.join(ir_navigator_runtime_dir,f), os.path.join(output_dir,f))
    

    # Templating
    with open(os.path.join(output_dir,'template.html'), 'r') as f:
        read_template = f.read()
    read_data = re.sub('INCLUDE_RI_HERE', read_data,read_template);
    
    
    
    #writing the output file
    output_file = ""
    current_suffix = ""
    while output_file=="":
        try_output_file = os.path.join(output_dir,m.name+str(current_suffix)+'.html')
        if not keep_existing or not os.path.exists(try_output_file):
            output_file=try_output_file
        else:
            if current_suffix == '': current_suffix=1
            else: current_suffix += 1 
        
    with open(output_file, 'w') as f:
        f.write(read_data)

    # Open a web browser if requested
    if openBrowser:
        webbrowser.open(output_file)
    
module.ir_navigator = ir_navigator