<style type="text/css">
	.boxpopup {font-family: Arial, sans-serif; font-size:90%; color:black; background:#DEEDF7;width:200px;text-align:center;padding:4px 5px 4px 5px;font-weight:bold;border:1px solid gray;}
        #pdqbox {position:absolute; visibility:hidden; z-index:200;}
</style>


<div id="pdqbox"></div>
        <script TYPE="text/javascript">
                var file_name;
                var modified;
                
                var OffsetX=-210;
                var OffsetY=140;
                var old, skn, iex = (document.all), yyy = -1000;
                var ns4 = document.layers;
                var ns6 = document.getElementById && !document.all;
                var ie4 = document.all;

                skn = document.getElementById('pdqbox').style;
                skn.visibility = "visible";
                skn.display = "none";

                document.onmousemove = get_mouse;

                function popup(msg){
                        var content = "<div class=boxpopup>" + msg + "</div>";
                        yyy = OffsetY;
                        if (ns4) {skn.document.write(content); skn.document.close(); skn.visibility="visible";}
                        if (ns6) {document.getElementById('pdqbox').innerHTML = content; skn.display=""; skn.visibility="visible"}
                        if (ie4) {document.all("pdqbox").innerHTML = content; skn.display=""}
                }

                function get_mouse(e) {
                        var x = (ns4 || ns6) ? e.pageX : event.x + document.body.scrollLeft;
                        skn.left = x + OffsetX;
                        var y = (ns4 || ns6) ? e.pageY : event.y + document.body.scrollTop;
                        skn.top = y + yyy;
                }

                function remove_popup(){
                        yyy=-1000;
                        if(ns4) {skn.visibility="hidden";}
                        else if (ns6 || ie4)
                                skn.display="none"
                }
        </script>


        return '\n'.join([ '<input onMouseOver="popup(\' %s \')" onMouseOut="remove_popup()" value="%s" type="submit"/><br/>' % (file(directory + f[ : f.rindex('.')] + '.txt').read().rstrip() if os.path.exists(directory + f[ : f.rindex('.')] + '.txt') else "", f) for f in os.listdir(directory) if os.path.isdir(f) == False and (f.endswith('.c') or f.endswith('.f'))])

