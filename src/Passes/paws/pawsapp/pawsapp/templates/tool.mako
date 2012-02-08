<%doc>
  Generic page for a tool
</%doc>


<%inherit file="base.mako"/>

<%namespace name="w"   file="lib/widgets.mako"/>
<%namespace name="adv" file="lib/advanced.mako"/>

<%def name="pagetitle()">
${info["title"]}
</%def>

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
  $(function () {
  $(".hero-unit").popover({html:true, placement: "bottom"});
  });
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
    ${w.icon("book", True)} Classic examples</button>
</label>

<form target="upload_target" action="${request.route_url('upload_user_file')}" method="post"
      enctype="multipart/form-data" id="upload_form" style="margin-bottom:-18px">
  <label for="pseudobutton">or from your own test cases:</label>
  <button id="pseudobutton" style="width:100%" class="btn btn-primary">
     ${w.icon("folder-open", True)} Local file(s)</button>
  <input type="file" id="upload_input" name="file" class="inp-hide"/><br/>
  <div id="pseudotextfile">&nbsp;</div>
</form>

<hr/>

<p>
  <button class="btn btn-primary" style="width:100%" id="run-button">
    ${w.icon("play", True)} Run</button>
</p>

<p>
  <button class="btn" style="width:100%" id="save-button">
    ${w.icon("download-alt")} Save Result</button><br/>
  <button class="btn" style="width:100%" id="print-button">
    ${w.icon("print")} Print Result</button>
</p>

<hr/>

## Mode toggle buttons
<div class="btn-group" data-toggle="buttons-radio" id="mode-buttons">
  <button class="btn" id="basic-button" style="width:50%">${w.icon("cog")} Basic</button>
  <button class="btn" id="adv-button"   style="width:50%">${w.icon("cog")} Adv.</button>
</div>

<form id="adv-form" class="${h.css_classes([('form-inline', True), ('hide', not advanced)])}"
      style="margin-bottom:0">

  <p></p>

  % if info["properties"]:
  <button class="btn btn-success" style="width:100%;text-align:left" data-toggle="modal" href="#adv-props-modal">
    ${w.icon("list-alt", True)} Properties</button><br/>
  % else:
  <div>${w.icon("list-alt")} No properties</div>
  % endif

  % if info["analyses"]:
  <button class="btn btn-success" style="width:100%;text-align:left" data-toggle="modal" href="#adv-analyses-modal">
    ${w.icon("list-alt", True)} Select Analyses</button><br/>
  % else:
  <div>${w.icon("list-alt")} No analyses</div>
  % endif

  % if info["phases"]:
  <button class="btn btn-success" style="width:100%;text-align:left" data-toggle="modal" href="#adv-phases-modal">
    ${w.icon("list-alt", True)} Phases</button>
  % else:
  <div>${w.icon("list-alt")} No phases</div>
  % endif

  ## "Properties" modal
  ${w.modal(u"Properties", advprop_body, id="adv-props-modal", icon="list-alt")}
  <%def name="advprop_body()">
  ${adv.properties_fields(info["properties"])}
  </%def>

  ## "Select Analyses" modal
  ${w.modal(u"Select Analyses", advanl_body, id="adv-analyses-modal", icon="list-alt")}
  <%def name="advanl_body()">
  ${adv.analyses_fields(info["analyses"])}
  </%def>

  ## "Phases" modal
  ${w.modal(u"Phases", advphases_body, id="adv-phases-modal", icon="list-alt")}
  <%def name="advphases_body()">
  ${adv.phases_fields(info["phases"])}
  </%def>

</form>

</%def>

## "No source" alert
<%def name="no_source_warning()">
<div class="alert alert-warning">
  ${w.icon("warning-sign")}
  <b>Source file(s) not found...</b> Please provide source code first.
</div>
</%def>



## MAIN COLUMN

<%def name="main_column()">

<div class="hero-unit" style="padding:.5em 1em; margin-bottom:1.5em"
     data-content="${info['descr']}" data-original-title="${tool}">
  <h2>
    ${h.image(request.static_url("pawsapp:static/img/favicon-trans.gif"), u"PAWS icon")}
    ${info["title"]}
  </h2>
</div>

## Tab headers
<ul id="op-tabs" class="nav nav-tabs noprint">
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
     ${self.no_source_warning()}
    </div>
  </div>
  ## Graph panel
  <div id="graph" class="tab-pane">
     ${self.no_source_warning()}
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
<div><a href="#" id="${ex}" class="btn" style="width:90%; text-align: left">
    ${w.icon("file")} ${ex}</a></div>
% endfor
</%def>


## Functions dialog

${w.modal(u"Select function to transform", functions_body, "choose-function-dialog")}

<%def name="functions_body()">
</%def>


</%def>
