import streamlit as st
from PIL import Image



def reptileVisualization():
    embedded_html = """<div class='tableauPlaceholder' id='viz1710691104915' style='position: relative'><noscript><a href='#'><img alt='Reptiles Found in India ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Re&#47;ReptilesfoundinIndia&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ReptilesfoundinIndia&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Re&#47;ReptilesfoundinIndia&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1710691104915');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""

    st.components.v1.html(embedded_html, height=500)

def mammals():
    embedded_html = """<div class='tableauPlaceholder' id='viz1712470748565' style='position: relative'><noscript><a href='#'><img alt='Mammals Found in India ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ma&#47;MammalsFoundinIndia&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='MammalsFoundinIndia&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ma&#47;MammalsFoundinIndia&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1712470748565');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    st.components.v1.html(embedded_html, height=500)

def amphibians():
   html= """<div class='tableauPlaceholder' id='viz1712470898006' style='position: relative'><noscript><a href='#'><img alt='Amphibians Found in India ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Am&#47;AmphibiansfoundinIndia&#47;Sheet2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='AmphibiansfoundinIndia&#47;Sheet2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Am&#47;AmphibiansfoundinIndia&#47;Sheet2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1712470898006');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
   st.components.v1.html(html, height=500)

def countryWise():
   html= """<div class='tableauPlaceholder' id='viz1712471024704' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Project-1countrywiseanimalspecies&#47;Sheet3&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='views&#47;Project-1countrywiseanimalspecies&#47;Sheet3?:language=en-GB&amp;:embed=true&amp;:sid=' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Project-1countrywiseanimalspecies&#47;Sheet3&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1712471024704');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
   st.components.v1.html(html, height=700)

def WildlifeAcrossIndia():
   html= """<div class='tableauPlaceholder' id='viz1712473011477' style='position: relative'><noscript><a href='#'><img alt='Sheet 2 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;K2&#47;K2NGG33NH&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;K2NGG33NH' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;K2&#47;K2NGG33NH&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1712473011477');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
   st.components.v1.html(html, height=500)

def CountryWise():
   html= """<div class='tableauPlaceholder' id='viz1712473248162' style='position: relative'><noscript><a href='#'><img alt='Sheet 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Projectcountrywiseanimalspecies&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Projectcountrywiseanimalspecies&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Projectcountrywiseanimalspecies&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1712473248162');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
   st.components.v1.html(html, height=500)

def CountryWiseThreatenedSpecies():
   html= """<div class='tableauPlaceholder' id='viz1712473188996' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Project-part2threatenedspecies&#47;Sheet6&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Project-part2threatenedspecies&#47;Sheet6' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Project-part2threatenedspecies&#47;Sheet6&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1712473188996');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
   st.components.v1.html(html, height=500)

def display_image(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=True)


def display_full_image(image_path):
    if st.button("Run"):
     countryWise()
     if st.button("Clear"):
      st.components.v1.html("", height=0)
    # with st.expander("Open Image", expanded=True):
    #     reptileVisualization()
    #     # st.image(image_path, use_column_width=True)
    #     if st.button("Close Image"):
    #         st.empty()

def viewer():
    button_labels = ['Countrywise Animals-Endangered Classification Found in India', 'Clear It', 'Wildlife Across India', 'Clear It', 'Reptiles Found in India', 'Clear It', 'Reptiles Found in India', 'Clear It', 'Mammals Found in India', 'Clear It', 'Countrywise Animal Species', 'Clear It', 'CountryWise-Threatened Species', 'Clear It', 'Amphibians Found in India', 'Clear It']

    for i in range(0, len(button_labels), 2):
        label = button_labels[i]
        clear_label = button_labels[i + 1]
        key = f'b{i // 2 + 1}'

        if st.button(label, key=key):
            if label.startswith("Mammals"):
                mammals()
            elif label.startswith("Amphibians"):
                amphibians()
            elif label.startswith("countrywise"):
                countryWise()
            elif label.startswith("Reptiles"):
                reptileVisualization()
            elif label.startswith("Wildlife Across India"):
                WildlifeAcrossIndia()
            elif label.startswith("Countrywise Animal Species"):
                CountryWise()
            elif label.startswith("CountryWise-Threatened Species"):
                CountryWiseThreatenedSpecies() 
            elif label.startswith("Countrywise Animals-Endangered Classification Found in India"):
                 CountryWiseThreatenedSpecies() 

            else:
                st.write("Invalid option")

            if st.button(clear_label, key=f'clear_{key}'):
                st.components.v1.html("", height=0)

    st.title("Data Visualization")


    # images = [
    #     "data/Reptiles.jpg",
    #    "data/Reptiles.jpg",
    #     "data/Reptiles.jpg",
    #    "data/Reptiles.jpg",
    #    "data/Reptiles.jpg",
    #     "data/Reptiles.jpg",
    #     "data/Reptiles.jpg",
    #     "data/Reptiles.jpg",
    #     "data/Reptiles.jpg",
    # ]

    # col1, col2, col3 = st.columns(3)

    # for i, image_path in enumerate(images):
    #     if i < 10:
    #         with col1:
    #             if st.button(f"Image {i+1}"):
    #                 display_full_image(image_path)
                
  
        #             # display_image(image_path)
        # elif 3 <= i < 6:
        #     with col2:
        #         if st.button(f"Image {i+1}"):
        #             reptileVisualization()
        #             if st.button("Clear Image"):
        #                  st.empty() 
        #             display_full_image(image_path)
        # else:
        #     with col3:
        #         if st.button(f"Image {i+1}"):
        #             reptileVisualization()
        #             if st.button("Clear Image"):
        #                  st.empty() 
        #             display_full_image(image_path)
        






# import streamlit as st

# # Page Configuration
# st.set_page_config(page_title="Visualization Concept", layout="wide")

# # Main Header
# st.title("Visualization Concept")

# # Main Box: Visualization
# st.header("Visualization")

# # Create a tree-like structure using expander widgets

# # Expander for Stats
# with st.expander("Stats"):
#     st.markdown("Display key statistical metrics with engaging visuals:")
#     st.image("1.jpg")  # Placeholder for a statistical visualization


# def run():
#     embedded_html = """
#     <div class='tableauPlaceholder' id='viz1708270298125' style='position: relative'><noscript><a href='#'><img alt='Reptiles Found in India ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Re&#47;ReptilesfoundinIndia&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ReptilesfoundinIndia&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Re&#47;ReptilesfoundinIndia&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>
#     <script type='text/javascript'>var divElement = document.getElementById('viz1708270298125');var vizElement = divElement.getElementsByTagName('object')[0];vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';var scriptElement = document.createElement('script');scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';vizElement.parentNode.insertBefore(scriptElement, vizElement);</script>
#     """

#     st.components.v1.html(embedded_html, height=700)

# run()




def explore():
    
    # st.subheader("Upload an animal image")
    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # if uploaded_file is not None:
    #     st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    #     st.write("Classifying...")

    #     # Ideally, here you'd use a model like ResNet or VGG for classification
    #     # status = classify_animal(uploaded_file)
    #     # Mocking the result for now
    #     st.write("This animal is:", "Endangered")  # Mock classification
    
    viewer()

# from prettymapp.geo import get_aoi
# from prettymapp.osm import get_osm_geometries
# from prettymapp.plotting import Plot
# from prettymapp.settings import STYLES

# aoi = get_aoi(address="Praça Ferreira do Amaral, Macau", radius=1100, rectangular=False)
# df = get_osm_geometries(aoi=aoi)

# fig = Plot(
#     df=df,
#     aoi_bounds=aoi.bounds,
#     draw_settings=STYLES["Peach"]
# ).plot_all()

# fig.savefig("map.jpg")

