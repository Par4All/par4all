<%doc>
  Generic page for a tool
</%doc>


<%inherit file="base.mako"/>

<%namespace name="w"   file="pawsapp:templates/lib/widgets.mako"/>
<%namespace name="adv" file="pawsapp:templates/lib/advanced.mako"/>

<%def name="css_slot()">
${h.stylesheet_link(request.static_url("pawsapp:static/css/jq/jquery-linedtextarea-min.css"), media="all")}
${h.stylesheet_link(request.static_url("pawsapp:static/css/jq/jquery.jqzoom-min.css"), media="all")}
${h.stylesheet_link(request.static_url("pawsapp:static/css/pygments-min.css"), media="all")}
</%def>

<%def name="js_slot()">
${h.javascript_link(request.route_url("routes.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/jquery-linedtextarea-min.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/jquery.jqzoom-core-pack-min.js"))}
<script type="text/javascript">
  operation = "${tool}";
  % if advanced:
  advanced = true;
  % endif
</script>
${h.javascript_link(request.static_url("pawsapp:static/js/init.js"))}
</%def>


## LEFT COLUMN

<%def name="left_column()">

<div style="text-align:right">
  <button class="btn btn-small" id="aminus">A-</button>
  <button class="btn btn-small" id="aplus">A+</button>
</div>

<h4 style="margin:.5em 0">Type or select source code from:</h4>

  <label>
    <button id="classic-button" style="width:100%" class="btn btn-primary"
	    data-toggle="modal" href="#classic-examples-dialog">
      <i class="icon-folder-open icon-white"></i> Classic examples</button>
  </label>

  <form target="upload_target" action="${request.route_url('upload_user_file')}"
	enctype="multipart/form-data" method="post" id="upload_form">
    <label for="pseudobutton">or from your own test cases:</label>
    <button id="pseudobutton" style="width:100%" class="btn btn-primary">
      <i class="icon-folder-open icon-white"></i> Browse</button>
    <input type="file" id="upload_input" name="file" class="inp-hide"/><br/>
    <div id="pseudotextfile">&nbsp;</div>
  </form>

<p>
  <button class="btn btn-primary disabled" style="width:100%" id="run-button">
    <i class="icon-play icon-white"></i> Run</button>
</p>
<p>
  <button class="btn disabled" style="width:100%" id="save-button">
    <i class="icon-download-alt"></i> Save Result</button><br/>
  <button class="btn disabled" style="width:100%" id="print-button">
    <i class="icon-print"></i> Print Result</button>
</p>

<form id="adv-form" class="${h.css_classes([('form-inline', True), ('hide', not advanced)])}">

  <h4>Advanced mode</h4>

  <button class="btn btn-success" style="width:100%" data-toggle="modal" href="#adv-props-modal">
    <i class="icon-cog icon-white"></i> Properties</button><br/>
  <button class="btn btn-success" style="width:100%" data-toggle="modal" href="#adv-analyses-modal">
    <i class="icon-cog icon-white"></i> Select Analyses</button><br/>
  <button class="btn btn-success" style="width:100%" data-toggle="modal" href="#adv-phases-modal">
    <i class="icon-cog icon-white"></i> Phases</button>
  
  ## "Properties" modal
  ${w.modal(u"Properties", advprop_body, id="adv-props-modal")}
  <%def name="advprop_body()">
  ${adv.properties_fields(props)}
  </%def>

  ## "Select Analyses" modal
  ${w.modal(u"Select Analyses", advanl_body, id="adv-analyses-modal")}
  <%def name="advanl_body()">
  ${adv.analyses_fields(analyses)}
  </%def>

  ## "Phases" modal
  ${w.modal(u"Phases", advphases_body, id="adv-phases-modal")}
  <%def name="advphases_body()">
  ${adv.phases_fields(phases)}
  </%def>

</form>

<div style="text-align: right; white-space:nowrap">
  Switch to ${h.link_to(u"basic mode" if advanced else u"advanced mode", url="#", id="adv-button")} »
</div>

</%def>


## MAIN COLUMN

<%def name="main_column()">

<iframe id="iframetoprint" style="height: 0px; width: 0px; position: absolute; -moz-opacity: 0; opacity: 0"></iframe>

<div class="hero-unit" style="padding:.5em 1em">
  <h2>${h.image(request.static_url("pawsapp:static/img/favicon-trans.gif"), u"PAWS icon")}
    ${descr}
    % if advanced:
    <span class="label label-success">advanced</span>
    % endif
  </h2>
</div>

## Tab headers
<ul id="op-tabs" class="nav nav-tabs">
  ${w.source_tab(id="1", active=True)}
  <li id="result_tab"><a href="#result" data-toggle="tab">${tool.upper()}</a></li>
  <li id="graph_tab"><a href="#graph" data-toggle="tab">GRAPH</a></li>
</ul>
  
## Tab panels
<div class="tab-content">
  ## Source code panel
  ${w.source_panel(id="1", active=True)}
  ## Result panel
  <div id="result" class="tab-pane">
    <div id="multiple-functions">
    </div>
    <div id="resultcode">
      Placeholder for results.
    </div>
  </div>
  ## Graph panel
  <div id="graph" class="tab-pane">
    Placeholder for dependence graphs.
  </div>
</div>

## Invisible iframe for file upload
<iframe id="upload_target" name="upload_target" class="hide"></iframe> 

<div id="source_tab_skel" class="hide">
  ${w.source_tab(id="__skel__")}
</div>
<div id="source_panel_skel" class="hide">
  ${w.source_panel(id="__skel__")}
</div>

</%def>


## DIALOG BOXES

<%def name="dialogs()">

## Classic examples modal

${w.modal(u"Please select an example:", classic_body, "classic-examples-dialog")}

<%def name="classic_body()">
% for ex in examples:
<div>${h.link_to(ex, id=ex, class_="btn small", style="width:90%")}</div>
% endfor
</%def>


## Functions dialog

${w.modal(u"Select function to transform", functions_body, "choose-function-dialog")}

<%def name="functions_body()">
</%def>


</%def>
