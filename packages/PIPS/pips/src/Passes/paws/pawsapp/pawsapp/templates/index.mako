<%doc>
  PAWS home page
</%doc>


<%inherit file="base.mako"/>


<%def name="pagetitle()">
Home Page
</%def>

<%def name="js_slot()">
<script type="text/javascript">
  $(function () {
% for s in sections:
% for t in s["tools"]:
    $("${'#%s-%s' % (s['path'], t['name'])}").popover({html:true, placement: "left"});
% endfor
% endfor
  });
</script>
</%def>


## LEFT COLUMN

<%def name="left_column()">
${text|n}
</%def>


## MAIN COLUMN

<%def name="main_column()">

<div class="hero-unit" style="padding:30px 30px 20px 30px">
  <h1>${h.image(request.static_url("pawsapp:static/img/paws-small-trans.gif"), u"PAWS Logo")} PIPS as a Web Service</h1>
  <p>Vestibulum id ligula porta felis euismod semper. Integer posuere
    erat a ante venenatis dapibus posuere velit aliquet. Duis mollis,
    est non commodo luctus, nisi erat porttitor ligula, eget lacinia
    odio sem nec elit.</p>
  <p><a class="btn btn-primary btn-large" href="http://pips4u.org" target="_blank">Learn more »</a></p>
</div>

<div class="row">
  <p class="span9 offset3">
    <span class="label label-info">Notice</span>
    Run the mouse over a heading to see the corresponding description.
  </p>
</div>

## Section
% for s in sections:

<hr/>

## Subsection
% for t in s["tools"]:
<% first = bool(s["tools"].index(t) == 0) %>
<div class="row">
  <div class="span3">
    % if first:
    <h3>${s["title"]}</h3>
    % else:
    &nbsp;
    % endif
  </div>
  <div class="span3">
    <h4 id="${s['path'] + '-' + t['name']}" title="${t['name'].upper()}" data-content="${t['descr']}">${t["name"].upper()}</h4>
  </div>
  <div class="span3">
    ${h.link_to(u"basic »", url="/%s/%s" % (s["path"], t["name"]), class_="btn btn-primary")}
    % if s["advmode"]:
    ${h.link_to(u"advanced »", url="/%s/%s/advanced" % (s["path"], t["name"]), class_="btn btn-success")}
    % endif
  </div>
</div>
% endfor

% if not s["tools"]:
<div class="row">
  <div class="span3">
    <h3>${s["title"]}</h3>
  </div>
</div>
% endif

% endfor
</%def>
