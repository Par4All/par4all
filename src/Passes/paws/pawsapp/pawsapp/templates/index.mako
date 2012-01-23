<%doc>
  Home page
</%doc>


<%inherit file="base.mako"/>


<%def name="pagetitle()">
PYPS DEMO PAGE
</%def>

<%def name="title()">
PYPS AS A WEB SERVICE
</%def>

<%def name="css_slot()">
<style>
  #main td { vertical-align: top }
  #left    { width: 25%; padding: 0 2em }
  #right   { padding: 1em }
  #sections .section { padding-bottom: 1em }
</style>
</%def>

<%def name="js_slot()">
<script type="text/javascript">
    $(function() {
% for s in c.sections:
	$("#${s['path']}").accordion({ 
	    active: false,
	    animated: 'bounceslide',
	    collapsible: true,
	    autoHeight: false
	});
% endfor
    });
</script>
</%def>


## Page content

<div id="main" class="ui-widget ui-widget-content ui-corner-all">
  <table>
    <tr>

      ## Left column

      <td id="left">
	<h4>${c.text|n}</h4>
      </td>

      ## Right column

      <td id="right">

	<table id="sections">

	  ## Section
	  % for s in c.sections:
	  <tr>
	    <td style="width:200px; white-space: nowrap">
	      <h2>${s["title"]}</h2>
	    </td>
	    <td style="width:100%" id="${s['path']}" class="section">

	      ## Subsection
	      % for t in s['tools']:
	      <h3>${h.link_to(t["name"].upper(), url="#")}</a></h3>
	      <div>
		<div>${t['descr']}</div>
		<p>
        	  <b>${h.link_to("basic", url="/%s/%s" % (s["path"], t["name"]))}</b>
		  % if s["advmode"]:
        	  <b>${h.link_to("advanced", url="/%s/%s/advanced" % (s["path"], t["name"]))}</b>
		  % endif
		</p>
	      </div>
	      % endfor

	    </td>
	  </tr>
	  % endfor

	</table>

      </td>
    </tr>
  </table>
</div>
