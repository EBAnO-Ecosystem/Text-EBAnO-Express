localExplanationReportTemplate = '''
<html>
<head>
  <meta content="text/html; charset=ISO-8859-1"
 http-equiv="content-type">
  <title>Local Explanation Report</title>
</head>
<style>
    .boxed {{
        border: 3px solid LightGray;
        padding: 10px ;
        margin: 10px ;
    }}
    
    #table_input {{
      width: 100%;
      background-color: #f1f1c1;
    }}
    
    #table_perturbation {{
      width: 100%;
      background-color: #247777;
      color: white;
    }}
    
    #positive_influential_color{{
        color: rgba(124, 252, 0, 1);
    }}
    
    #negative_influential_color{{
        color: rgba(255, 99, 71, 1);
    }}
    
    #neutral_influential_color{{
        color: rgba(255, 255, 0, 1);
    }}
    
    .featureBoxed{{
        border: 1px solid LightGray;
        padding: 10px ;
        padding-left: 30px ;
        padding-right: 30px ;
        margin: 30px ;
        
    }}
    
    /* Style the tab */
    .tab {{
      overflow: hidden;
      border: 1px solid #ccc;
      background-color: #f1f1f1;
    }}
    
    /* Style the buttons that are used to open the tab content */
    .tab button {{
      background-color: inherit;
      float: left;
      border: none;
      outline: none;
      cursor: pointer;
      padding: 14px 16px;
      transition: 0.3s;
    }}
    
    /* Change background color of buttons on hover */
    .tab button:hover {{
      background-color: #ddd;
    }}
    
    /* Create an active/current tablink class */
    .tab button.active {{
      background-color: #ccc;
    }}
    
    /* Style the tab content */
    .tabcontent {{
      display: none;
      padding: 6px 12px;
      border: 1px solid #ccc;
      border-top: none;
    }}
    
</style>
<body>
    <!-- Tab links -->
    <div class="tab">
      <button class="tablinks" onclick="openFeatureType(event, 'MLWE')" id="defaultOpen">MLWE</button>
      <button class="tablinks" onclick="openFeatureType(event, 'POS')">POS</button>
      <button class="tablinks" onclick="openFeatureType(event, 'SEN')">SEN</button>
      <button class="tablinks" onclick="openFeatureType(event, 'RND')">RND</button>
    </div>
    
    <!-- Tab content -->
    <div id="MLWE" class="tabcontent">
      <h1>INPUT INFO</h1>
      <h3>Original Raw Text</h3>
      <div class="boxed">{raw_text}</div>
      <h3>Original Cleaned Text</h3>
      <div class="boxed">{clean_text}</div>
      <h3>Original Preprocessed Text</h3>
      <div class="boxed">{pre_text}</div>
      <div id="input_info">
      <table id="table_input">
          <tr>
            <th>Original Probabilities</th>
            <th>Original Label</th>
            <th>Label Name</th>
          </tr>
          <tr>
            <th>{original_probabilities}</th>
            <th>{original_label}</th>
            <th>{label_name}</th>
          </tr>
      </table>
    </div>
      <hr>
      <h1>MLWE Summary</h1>
      <div class="boxed">{html_mlwe_summary}</div>
      <hr>
      <h1>MLWE Ordered Features</h1>
      <div>{mlwe_local_explanations}</div>
    </div>

    <div id="POS" class="tabcontent">
      <h1>INPUT INFO</h1>
      <h3>Original Raw Text</h3>
      <div class="boxed">{raw_text}</div>
      <h3>Original Cleaned Text</h3>
      <div class="boxed">{clean_text}</div>
      <h3>Original Preprocessed Text</h3>
      <div class="boxed">{pre_text}</div>
      <div id="input_info">
      <table id="table_input">
          <tr>
            <th>Original Probabilities</th>
            <th>Original Label</th>
            <th>Label Name</th>
          </tr>
          <tr>
            <th>{original_probabilities}</th>
            <th>{original_label}</th>
            <th>{label_name}</th>
          </tr>
      </table>
    </div>
      <hr>
      <h1>POS Summary</h1>
      <div class="boxed">{html_pos_summary}</div>
      <hr>
      <h1>POS Ordered Features</h1>
      <div>{pos_local_explanations}</div>
    </div>

    <div id="SEN" class="tabcontent">
      <h1>INPUT INFO</h1>
      <h3>Original Raw Text</h3>
      <div class="boxed">{raw_text}</div>
      <h3>Original Cleaned Text</h3>
      <div class="boxed">{clean_text}</div>
      <h3>Original Preprocessed Text</h3>
      <div class="boxed">{pre_text}</div>
      <div id="input_info">
      <table id="table_input">
          <tr>
            <th>Original Probabilities</th>
            <th>Original Label</th>
            <th>Label Name</th>
          </tr>
          <tr>
            <th>{original_probabilities}</th>
            <th>{original_label}</th>
            <th>{label_name}</th>
          </tr>
      </table>
    </div>
      <hr>
      <h1>SEN Summary</h1>
      <div class="boxed">{html_sen_summary}</div>
      <hr>
      <h1>SEN Ordered Features</h1>
      <div>{sen_local_explanations}</div>
    </div>
    
    <div id="RND" class="tabcontent">
      <h1>INPUT INFO</h1>
      <h3>Original Raw Text</h3>
      <div class="boxed">{raw_text}</div>
      <h3>Original Cleaned Text</h3>
      <div class="boxed">{clean_text}</div>
      <h3>Original Preprocessed Text</h3>
      <div class="boxed">{pre_text}</div>
      <div id="input_info">
      <table id="table_input">
          <tr>
            <th>Original Probabilities</th>
            <th>Original Label</th>
            <th>Label Name</th>
          </tr>
          <tr>
            <th>{original_probabilities}</th>
            <th>{original_label}</th>
            <th>{label_name}</th>
          </tr>
      </table>
    </div>
      <hr>
      <h1>RND Summary</h1>
      <div class="boxed">{html_rnd_summary}</div>
      <hr>
      <h1>RND Ordered Features</h1>
      <div>{rnd_local_explanations}</div>
    </div>

<script>
function openFeatureType(evt, featureType) {{
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {{
    tabcontent[i].style.display = "none";
  }}
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {{
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }}
  document.getElementById(featureType).style.display = "block";
  evt.currentTarget.className += " active";
}}
 
// Get the element with id="defaultOpen" and click on it
document.getElementById("defaultOpen").click();

</script>
</body>
</html>'''
