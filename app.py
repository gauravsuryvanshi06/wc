
import explore, upload, counting
from streamlit_option_menu import option_menu

from io import StringIO
import streamlit as st
from classification import animal_classification
from counting import animal_counting

from upload import uploads
from information import animal_information, robo
from explore import display_image, explore, viewer
import pandas as pd

import numpy as np

from PIL import Image

# import e as TSC

import streamlit.components.v1 as components


def display_homepage():
    """

    Display homepage with a compelling slogan and introduction.
    """
    # st.markdown(
    #      f"""
    #      <style>
    #      .stApp {{
    #          background: url("https://happydays365.org/wp-content/uploads/2019/12/World-Wildlife-Conservation-Day-980x653.jpg");
    #          background-size: coverx
    #      }}
    #      </style>
    #      """,
    #      unsafe_allow_html=True
    #  )

    st.image("data/elephant-48423.svg", use_column_width=True)  # Add an attractive wildlife image
    st.markdown("<h1 style='text-align: lcenter; color: green;'>Wildlife Conservation</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Join hands in making the world a sanctuary for wildlife.</h3>", unsafe_allow_html=True)

    video_file = open('data/1.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    st.write("""
 

Welcome to our Wildlife Conservation Web Application.
               We're using state-of-the-art technology to drive the mission of Wildlife Conservation.

 
🌿Count with Precision: Our tech doesn't just count animals; it spotlights each one uniquely. 🐾✨
Computer do the Tagging

🔍Classify, Track, Preserve: Beyond counting, this project also classifies and tracks, capturing the essence of conservation. An interface where nature and technology dance in harmony. 🌿🤖
             
             
Anyone collecting camera trap photos can upload them to Wildlife Project. Photos are stored online so you can access them from anywhere, from any device or computer, even out in the field.

Animals in your photos are automatically identified using machine learning technology. Thousands of images can be tagged within minutes, saving you time to do the important work.

    """)
    
    # with st.status("Downloading data...", expanded=True) as status:
    #     st.write("Searching for data...")
    #     time.sleep(2)

    #     st.write("Found URL.")
    #     time.sleep(1)
    #     st.write("Downloading data...")
    #     time.sleep(1)
    #     status.update(label="Download complete!", state="complete", expanded=False)

st.button('Rerun')













# import streamlit as st



# import home, trending, account, your, about
# st.set_page_config(
#         page_title="Pondering",
# )



# class MultiApp:

#     def __init__(self):
#         self.apps = []

#     def add_app(self, title, func):

#         self.apps.append({
#             "title": title,
#             "function": func
#         })

#     def run():
#         # app = st.sidebar(
#         with st.sidebar:        
#             app = option_menu(
#                 menu_title='Pondering ',
#                 options=['Home','Account','Trending','Your Posts','about'],
#                 icons=['house-fill','person-circle','trophy-fill','chat-fill','info-circle-fill'],
#                 menu_icon='chat-text-fill',
#                 default_index=1,
#                 styles={
#                     "container": {"padding": "5!important","background-color":'black'},
#         "icon": {"color": "white", "font-size": "23px"}, 
#         "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
#         "nav-link-selected": {"background-color": "#02ab21"},}
                
#                 )

        
#         if app == "Home":
#             home.app()
#         if app == "Account":
#             account.app()    
#         if app == "Trending":
#             trending.app()        
#         if app == 'Your Posts':
#             your.app()
#         if app == 'about':
#             about.app()    
             
          
             
#     run()















    
              
# def main():
#     st.title("Wildlife Conservation")


#     st.sidebar.title("Menu")

#     choice = st.sidebar.radio("Select", ["Home","Explore Data","Upload Images", "Animal Counting", "Animal Classification", "Animal Information"])

#     if choice == "Home":
#         display_homepage()

#     elif choice == "Explore Data":
#         st.header("Exploe Data Visulaization")
#         explore()

#     elif choice == "Upload Images":
#         st.header("Upload Images")
#         upload()

#     elif choice == "Animal Information":
#         st.header("Animal Information")
#         animal_information()
#     elif choice == "Animal Counting":
#         st.header("Animal Counting")
#         animal_counting()

#     elif choice == "Animal Classification":
#         st.header("Animal Classification")
#         animal_classification()

#     elif choice == "Animal Information":
#         st.header("Animal Information")
#         animal_information()


def main():
    # st.title("Wildlife Conservation")
    st.sidebar.title("")

    with st.sidebar:        
        app = option_menu(
            menu_title='Wildlife Insights ',
            options=['Home','Data Visualization','Upload Images','Animal Counting','Animal Classification','Animal Information'],
            icons=['house-fill','chart-line','cloud-upload','person-badge','grid','info-circle'],
            menu_icon='chat-text-fill',
            default_index=0,
            styles={
                "container": {"padding": "5px","background-color":''},
                "icon": {"color": "black", "font-size": "23px"}, 
                "nav-link": {"color":"black","font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "Green"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )

    if app == "Home":
        display_homepage()

    elif app == "Data Visualization":
        viewer()

    elif app == "Upload Images":
        uploads()

    elif app == "Animal Counting":
        animal_counting()

    elif app == "Animal Classification":
        animal_classification()

    elif app == "Animal Information":
        robo()
        animal_information()


    
    st.image("data/Identify.png", use_column_width=True) 

    paragraph = """
    We urgently need to understand the presence and absence of species and their population trends in the areas where we work. Camera trap surveys are an effective method for doing this, but each camera trap survey produces tens of thousands of images, all of which need to be looked at. 
    but this can involve weeks or even months of painstaking effort. To speed up this process, this web application is crowd-sourcing the identifications through our Instant Wild App. Another method that has huge potential to transform our work is automated image recognition, using the power of machine learning to complete large scale species recognition within images.   
    By using the essentially limitless processing power of computers to do the hard work of identifying species within images, we can dramatically speed up the analysis of camera trap surveys and monitor wildlife trends in near real time. 


    """


    paraQ = """
    Is the image empty? 

    Is there a human in the image? 

    What species group (i.e. Animal or  bird) are present in the image? 

    What specific species are present in the image? 

    How many animals are present in the image? 

    """

    paraC = """


    ⦿ Expertise needed for training and testing algorithms. 

    ⦿ Availability of large, manually identified training datasets.

    ⦿ Complex data preparation requirements.

    ⦿ Barriers for implementing algorithms in the field.

    ⦿ Limited adoption due to complexity and accessibility issues.

    """






    st.markdown("<h2 style='text-align: center;'>Why is machine learning important for  wildlife conservation?</h2>", unsafe_allow_html=True)
    st.write(paragraph)


    st.markdown("<h2 style='text-align: center;'>Key questions of focus include: </h2>", unsafe_allow_html=True)


    st.write(paraQ)


    st.markdown("<h2 style='text-align: center;'>Challenges of machine learning in wildlife conservation:</h2>", unsafe_allow_html=True)

    st.write(paraC)


    
    # st.write ("See Map below for the location of the animals")
    # df = pd.DataFrame({
    # "col1": np.random.randn(1000) / 50 + 37.76,
    # "col2": np.random.randn(1000) / 50 + -122.4,
    # "col3": np.random.randn(1000) * 100,
    # "col4": np.random.rand(1000, 4).tolist(),})

    # st.map(df,
    # latitude='col1',
    # longitude='col2',
    # size='col3',
    # color='col4')

if __name__ == "__main__":
    main()









# data = pd.read_csv('data/animal.csv')



# st.title('Animal Population Information')

# countries = data['Country'].unique()
# selected_country = st.selectbox('Select a Country', countries)

# if selected_country:
#     st.subheader(f'Animal Population in {selected_country}')
#     country_data = data[data['Country'] == selected_country]
#     st.write(country_data)



# # Set up connection.
# tableau_auth = TSC.PersonalAccessTokenAuth(
#     st.secrets["tableau"]["token_name"],
#     st.secrets["tableau"]["token_secret"],
#     st.secrets["tableau"]["site_id"],
# )
# server = TSC.Server(st.secrets["tableau"]["server_url"], use_server_version=True)


# Get various data.
# Explore the tableauserverclient library for more options.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
# @st.cache_data(ttl=600)
# def run_query():
#     with server.auth.sign_in(tableau_auth):

#         # Get all workbooks.
#         workbooks, pagination_item = server.workbooks.get()
#         workbooks_names = [w.name for w in workbooks]

#         # Get views for first workbook.
#         server.workbooks.populate_views(workbooks[0])
#         views_names = [v.name for v in workbooks[0].views]

        # # Get image & CSV for first view of first workbook.
        # view_item = workbooks[0].views[0]
        # server.views.populate_image(view_item)
        # server.views.populate_csv(view_item)
        # view_name = view_item.name
#         # view_image = view_item.image
#         # `view_item.csv` is a list of binary objects, convert to str.
#         view_csv = b"".join(view_item.csv).decode("utf-8")

#         return workbooks_names, views_names, view_name, view_image, view_csv

# workbooks_names, views_names, view_name, view_image, view_csv = run_query()


# st.subheader("📊 Data")
# st.write(f"And here's the data for view *{view_name}*:")
# st.write(pd.read_csv(StringIO(view_csv)))







# import streamlit as st
# from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
# from htbuilder.units import percent, px
# from htbuilder.funcs import rgba, rgb


# def image(src_as_string, **style):
#     return img(src=src_as_string, style=styles(**style))

# def vimeo_video(video_id):
#     return f'''
#         <iframe src="//player.vimeo.com/video/{video_id}?background=1" 
#                 frameborder="0" 
#                 webkitallowfullscreen 
#                 mozallowfullscreen 
#                 allowfullscreen 
#                 style="width: 944px; height: 530.528px;">
#         </iframe>
#     '''

# def main():
#     st.markdown(vimeo_video('303126374'), unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()


# def link(link, text, **style):
#     return a(_href=link, _target="_blank", style=styles(**style))(text)


# def layout(*args):

#     style = """
#     <style>
#       # MainMenu {visibility: hidden;}
#       footer {visibility: hidden;}
#      .stApp { bottom: 105px; }
#     </style>
#     """

#     style_div = styles(
#         position="fixed",
#         left=0,
#         bottom=0,
#         margin=px(0, 0, 0, 0),
#         width=percent(100),
#         color="black",
#         text_align="center",
#         height="auto",
#         opacity=1
#     )

#     style_hr = styles(
#         display="block",
#         margin=px(8, 8, "auto", "auto"),
#         border_style="inset",
#         border_width=px(2)
#     )

#     body = p()
#     foot = div(
#         style=style_div
#     )(
#         hr(
#             style=style_hr
#         ),
#         body
#     )

#     st.markdown(style, unsafe_allow_html=True)

#     for arg in args:
#         if isinstance(arg, str):
#             body(arg)

#         elif isinstance(arg, HtmlElement):
#             body(arg)

#     st.markdown(str(foot), unsafe_allow_html=True)


# def footer():
#     myargs = [
#         "Made in ",image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',width=px(25), height=px(25)),
#         " with ❤️ by ",
#         link("https://twitter.com/ChristianKlose3", "@ChristianKlose3"),
#         br(),
#         link("https://buymeacoffee.com/chrischross", image('https://i.imgur.com/thJhzOO.png')),
#     ]
#     layout(*myargs)


# if __name__ == "__main__":
#     footer()




st.image("data/9.png", use_column_width=True)  


st.write("-------")
st.write("## Wildlife Conservation ML Project")
st.write("Keep in touch with us")
st.write("Sign up for wildlife updates and ways to get involved") 
st.button("SIGN UP")

st.write("### Guided by : Dr. Mahendra B. Salunke ")
st.write("###### Developed by: Group No. 9")
st.write("- ###### Vedant Angane")
st.write("- ###### Manavi Gawande")
st.write("- ###### Vaidahi Pawar")
st.write("- ###### Gaurav Suryavanshi")
# Copyright notice
st.write("#### Copyright © 2024. All rights reserved.")

st.write("---")
