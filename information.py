# import base64
# import streamlit as st
# import pandas as pd

# from classification import display_text_result, perform_inference

# import streamlit as st


# import streamlit as st
# import requests
# from io import BytesIO
# from PIL import Image
# import base64
# import json



# # from inference_sdk import InferenceHTTPClient

# # CLIENT = InferenceHTTPClient(
# #     api_url="https://detect.roboflow.com",
# #     api_key="coZNWQb0K35yZfbOU3rf"
# # )
# # file1 = st.file_uploader("Upload File")
# # result = CLIENT.infer(file1, model_id="animal-detection-jvsw5/1")
# # from roboflow import Roboflow

# # rf = Roboflow(api_key="vlmLfp4rGVbfbzmqp8zx")
# # project = rf.workspace().project("animal-detection-jvsw5")
# # model = project.version("1").model

# # job_id, signed_url, expire_time = model.predict_video(
# #     "Trail Cam Videos_ 1000 Hours in the Jungle_ Rain Forest Animals Puma.mp4",
# #     fps=5,
# #     prediction_type="batch-video",
# # )

# # results = model.poll_until_video_results(job_id)

# # print(results)
# # import streamlit as st
# # import streamlit as st
# # from inference_sdk import InferenceHTTPClient
# # import base64

# # import base64
# # # from roboflow import InferenceHTTPClient
# # import streamlit as st



import time
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def animal_classification():
    model = tf.keras.models.load_model('data/1.h5') 
    class_labels = ["Least Concern", "Near Threatened", "Vulnerable","Endangered", ...] 
    def preprocess_image(image):
        img = image.resize((299, 299)) 
        img = img.convert('RGB') 
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(image):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        st.write(class_labels[predicted_class],   "Animal      , " , prediction[0][predicted_class])
        return prediction

    st.title('Image Classifier')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict(image)

        st.write("Prediction:")
        st.write(prediction)

from io import BytesIO
from PIL import Image
import base64

import streamlit as st
import streamlit as st
# from inference_sdk import InferenceHTTPClient
import base64
import pandas as pd

import base64
import streamlit as st

def display_text_result(result):
    time_taken = result.get("time")
    image_width = result.get("image", {}).get("width")
    image_height = result.get("image", {}).get("height")
    predictions = result.get("predictions", [])
    animal_names = [prediction.get('class') for prediction in predictions]

    st.write("#### Detected Animals List :")

    for i, animal_name in enumerate(animal_names, start=1):
        st.write(f"#### Animal {i}:  {animal_name}")
    # for i in range(5):
    #     if st.checkbox("", key=f'uniqueKey{i+4}', value=False):
    #         pass
    st.write("All detected animals:", ", ".join(animal_names))

    result_text = f"Inference Time: {time_taken} seconds \n \n Image Width: {image_width}, Image Height: {image_height}\n"
    for i, prediction in enumerate(predictions, start=1):
        result_text += (f"\nPrediction {i}:\n"
                        f"Class: {prediction.get('class')}\n \n"
                        f"Confidence: {prediction.get('confidence')}\n"
                        f"\nPosition: (x: {prediction.get('x')}, y: {prediction.get('y')})\n"
                        f"\nSize: (width: {prediction.get('width')}, height: {prediction.get('height')})\n")

    st.write(result_text)
    return animal_names

def robo():
    st.title("Image Inference")
    detected_animals = display_text_result(result)  # This will return the list of animal names
    for animal in detected_animals:
      st.markdown(f"### Information for {animal.capitalize()}:")
      animal_info = fetch_animal_info_from_dataset(animal)
      if animal_info is not None and not animal_info.empty:
        st.markdown("**Height:** " + str(animal_info['Height (cm)']))
        st.markdown("**Weight:** " + str(animal_info['Weight (kg)']))
        st.markdown("**Color:** " + animal_info['Color'])
        st.markdown("**Lifespan:** " + str(animal_info['Lifespan (years)']))
        
        st.write("Diet:", animal_info['Diet'])
        st.write("Habitat:", animal_info['Habitat'])
        st.write("Predators:", animal_info['Predators'])
        st.write("Average Speed (km/h):", str(animal_info['Average Speed (km/h)']))
        st.write("Countries Found:", animal_info['Countries Found'])
        
        st.write("Conservation Status:")
        display_conservation_status(animal_info['Conservation Status'])
        
        st.write("Family:", animal_info['Family'])
        st.write("Gestation Period (days):", str(animal_info['Gestation Period (days)']))
        st.write("Top Speed (km/h):", str(animal_info['Top Speed (km/h)']))
        st.write("Social Structure:", animal_info['Social Structure'])
        st.write("Offspring per Birth:", str(animal_info['Offspring per Birth']))
      else:
        st.write("No information found for this animal.")

    """
Discover detailed insights about animals with our enhanced visual presentation feature. Dive into our comprehensive dataset to learn more about your favorite animals, displayed in a visually appealing format.

"""    
    if st.checkbox('Show the entire dataset',key='1234'):
      st.dataframe(animal_data)
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="coZNWQb0K35yZfbOU3rf"
    )

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.markdown("<h3 style='text-align: left; color: black;'>Classifying...</h3>", unsafe_allow_html=True)

        # Create a progress bar instance
        progress_bar = st.progress(0)

        # Time duration for the progress bar to fill
        duration = 5  # duration in seconds
        increments = 20  # increase the progress bar in 5% increments
        percent_complete = st.empty()


        for i in range(increments + 1):
            # Update progress bar by increment

            progress = int(100 * i / increments)
            progress_bar.progress(progress)
            percent_complete.text(f"Progress: {progress}%")
            # Wait a bit before the next update
            time.sleep(duration / increments)

        image_bytes = uploaded_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        result = CLIENT.infer(image_base64, model_id="animal-detection-jvsw5/1")

        st.write("Inference result:")
        display_text_result(result)



animal_data = pd.read_csv('data/AnimalDataset.csv')

def fetch_animal_info_from_dataset(animal_name):
    """
    Fetch animal information from the provided dataset for names up to three words long.
    """
    animal_name = animal_name.lower().strip().split()
    if len(animal_name) == 1:
        info = animal_data[animal_data['Animal'].str.lower() == animal_name[0]].squeeze()
    else:
        search_query = " ".join(animal_name)  # Re-join for exact match in case of multi-word names
        info = animal_data[animal_data['Animal'].str.lower() == search_query].squeeze()
    return info if not info.empty else None

def display_conservation_status(status):
    """
    Display conservation status with color coding.
    """


    status_colors = {

        'Least Concern': 'green',

        'Vulnerable': 'orange',

        'Endangered': 'red',

        'Critically Endangered': 'darkred'

    }

    color = status_colors.get(status, 'blue')
    st.markdown(f"<div style='color: {color}; font-size: 24px;'>{status}</div>", unsafe_allow_html=True)

def animal_information():
    
    user_input = st.text_input("", "", key="animal_search_input").strip().lower()
    
    if user_input:
        matches = animal_data[animal_data['Animal'].str.lower().str.contains(user_input)]['Animal'].unique()
        
        if len(matches) > 0:
          selected_animal = st.selectbox("Select an animal:", matches, key="animal_select_box")
          info = fetch_animal_info_from_dataset(selected_animal)
            
          if info is not None:
            st.markdown("**Height:** " + str(info['Height (cm)']))
            st.markdown("**Weight:** " + str(info['Weight (kg)']))
            st.markdown("**Color:** " + info['Color'])
            st.markdown("**Lifespan:** " + str(info['Lifespan (years)']))
            
            st.write("Diet:", info['Diet'])
            st.write("Habitat:", info['Habitat'])
            st.write("Predators:", info['Predators'])
            st.write("Average Speed (km/h):", str(info['Average Speed (km/h)']))
            st.write("Countries Found:", info['Countries Found'])
            
            st.write("Conservation Status:")
            display_conservation_status(info['Conservation Status'])
            
            st.write("Family:", info['Family'])
            st.write("Gestation Period (days):", str(info['Gestation Period (days)']))
            st.write("Top Speed (km/h):", str(info['Top Speed (km/h)']))
            st.write("Social Structure:", info['Social Structure'])
            st.write("Offspring per Birth:", str(info['Offspring per Birth']))
          else:
            st.write("No information found for this animal.")
            

def robo():
    st.title("Image Inference")

    if st.checkbox('Show the entire dataset', key='1234'):
        st.dataframe(animal_data)
    
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="coZNWQb0K35yZfbOU3rf"
    )

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.markdown("<h5 style='text-align: left; color: black;'>Classifying...</h3>", unsafe_allow_html=True)
        
        colors = [
    "#cefa",  # assuming the correct color code should have two additional characters
    "#abf7b1", "#83f28f", "#5ced73", 
    "#39e75f", "#1fd655", "#00c04b", 
    "#00ab41", "#008631"
]
        # Create a container for better control over layout
        container = st.container()
        progress_text = container.empty()  # Placeholder for displaying progress text
        my_bar = st.progress(0)            # Initial progress bar setup

        # Display initial operation message
        progress_text.markdown("<h3 style='text-align: center; color: black;'>Operation in progress. Please wait.</h3>", unsafe_allow_html=True)

        # Time duration for the progress bar to fill
        duration = 5  # duration in seconds
        increments = 100  # increment for finer progress control

        for i in range(increments + 1):
            # Calculate progress percentage
            percent_complete = int(100 * i / increments)
            if i==101:
                st.markdown("<h5 style='text-align: left; color: black;'>Performing inference...</h3>", unsafe_allow_html=True)

            # Determine color based on progress
            color_index = min(len(colors) - 1, percent_complete // (100 // len(colors)))  # Ensure index is within range
            color = colors[color_index]

            # Update progress bar and text with color
            my_bar.progress(percent_complete)
            progress_text.markdown(f"<h3 style='text-align: center; color: {color};'>{percent_complete}% - Operation in progress...</h3>", unsafe_allow_html=True)
            
            # Pause briefly
            time.sleep(duration / increments)

        # # Clear the progress bar and text after completion
        # my_bar.empty()
        # progress_text.empty()

        # Display final message and rerun button


        
        # colors = [
        #     "#cefa",  # corrected assuming full color code needed
        #     "#abf7b1", "#83f28f", "#5ced73",
        #     "#39e75f", "#1fd655", "#00c04b",
        #     "#00ab41", "#008631"
        # ]

        # # CSS to dynamically change the progress bar color
        # # Corrected CSS to dynamically change the progress bar color
        # progress_bar_css = """
        # <style>
        #     .stProgress > div > div > div > div {{
        #         background-color: {color} !important;
        #     }}
        #     /* Reduce excess spacing */
        #     .stProgress > div {{
        #         padding: 0;
        #         margin: 0;
        #     }}
        # </style>
        # """
        # container = st.container()
        # progress_text = container.empty()  # Placeholder for displaying progress text


        # # Container for the progress bar
        # progress_container = st.empty()
        # my_bar = progress_container.progress(0)

        # # Display initial operation message
        # st.markdown("<h3 style='text-align: center; color: black;'>Operation in progress. Please wait.</h3>", unsafe_allow_html=True)

        # # Time duration for the progress bar to fill
        # duration = 5  # duration in seconds
        # increments = 100  # increment for finer progress control

        # for i in range(increments + 1):
        #     # Calculate progress percentage
        #     percent_complete = int(100 * i / increments)
            
        #     # Determine color based on progress
        #     color_index = min(len(colors) - 1, percent_complete // (100 // len(colors)))  # Ensure index is within range
        #     color = colors[color_index]
        #     progress_text.markdown(f"<h5 style='text-align: center; color: {color};'>{percent_complete}% - Operation in progress...</h5>", unsafe_allow_html=True)

        #     # Update progress bar color via CSS
        #     st.markdown(progress_bar_css.format(color=color), unsafe_allow_html=True)
        #     # Update progress bar
        #     my_bar.progress(percent_complete)
            
        #     # Pause briefly
        #     time.sleep(duration / increments)
            
        image_bytes = uploaded_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        result = CLIENT.infer(image_base64, model_id="animal-detection-jvsw5/1")
        # st.write("### Inference result:")
        
        detected_animals = display_text_result(result)  # Ensures result is defined before use
        st.write(detected_animals[0])
        i=0
        for index, animal in enumerate(detected_animals):
            user_input = animal
            if user_input:
                matches = animal_data[animal_data['Animal'].str.lower().str.contains(user_input)]['Animal'].unique()
                
                if len(matches) > 0:
                    key = f"animal_select_box_{index}"  # Unique key based on index
                    selected_animal = st.selectbox("Select an animal:", matches, key=key)
                info = fetch_animal_info_from_dataset(selected_animal)
                    
                if info is not None:
                    st.markdown("**Height:** " + str(info['Height (cm)']))
                    st.markdown("**Weight:** " + str(info['Weight (kg)']))
                    st.markdown("**Color:** " + info['Color'])
                    st.markdown("**Lifespan:** " + str(info['Lifespan (years)']))
                    
                    st.write("Diet:", info['Diet'])
                    st.write("Habitat:", info['Habitat'])
                    st.write("Predators:", info['Predators'])
                    st.write("Average Speed (km/h):", str(info['Average Speed (km/h)']))
                    st.write("Countries Found:", info['Countries Found'])
                    
                    st.write("Conservation Status:")
                    display_conservation_status(info['Conservation Status'])
                    
                    st.write("Family:", info['Family'])
                    st.write("Gestation Period (days):", str(info['Gestation Period (days)']))
                    st.write("Top Speed (km/h):", str(info['Top Speed (km/h)']))
                    st.write("Social Structure:", info['Social Structure'])
                    st.write("Offspring per Birth:", str(info['Offspring per Birth']))
                else:
                    st.write("No information found for this animal.")

            # st.markdown(f"Information for {animal.capitalize()}:")
            # animal_info = fetch_animal_info_from_dataset(animal)
            # st.markdown(f"Information for {animal.capitalize()}:")
            # if animal_info is not None and not animal_info.empty:
            #         st.markdown("**Height:** " + str(animal_info['Height (cm)']))
            #         st.markdown("**Weight:** " + str(animal_info['Weight (kg)']))
            #         st.markdown("**Color:** " + animal_info['Color'])
            #         st.markdown("**Lifespan:** " + str(animal_info['Lifespan (years)']))
                    
            #         st.write("Diet:", animal_info['Diet'])
            #         st.write("Habitat:", animal_info['Habitat'])
            #         st.write("Predators:", animal_info['Predators'])
            #         st.write("Average Speed (km/h):", str(animal_info['Average Speed (km/h)']))
            #         st.write("Countries Found:", animal_info['Countries Found'])
                    
            #         st.write("Conservation Status:")
            #         display_conservation_status(animal_info['Conservation Status'])
                    
            #         st.write("Family:", animal_info['Family'])
            #         st.write("Gestation Period (days):", str(animal_info['Gestation Period (days)']))
            #         st.write("Top Speed (km/h):", str(animal_info['Top Speed (km/h)']))
            #         st.write("Social Structure:", animal_info['Social Structure'])
            #         st.write("Offspring per Birth:", str(animal_info['Offspring per Birth']))

            # else:
            #             st.write("No information found for this animal.")
    else:
            st.write("Please upload an image to start the inference process.")

# Ensure to call robo() function at the appropriate place in your script
# robo()





# # st.title("Image Inference")

# # # Initialize InferenceHTTPClient
# # CLIENT = InferenceHTTPClient(
# #     api_url="https://detect.roboflow.com",
# #     api_key="coZNWQb0K35yZfbOU3rf"
# # )

# # # File uploader
# # uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# # # Perform inference if file is uploaded
# # if uploaded_file is not None:
# #     st.write("Performing inference...")
    
# #     # Convert file to bytes
# #     image_bytes = uploaded_file.read()

# #     # Encode image bytes to base64
# #     image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# #     # Perform inference
# #     result = CLIENT.infer(image_base64, model_id="animal-detection-jvsw5/1")

# #     # Display result in text format
# #     st.write("Inference result:")

# #     display_text_result(result)

# # def display_text_result(result):
# #     # Parse the JSON result
# #     time_taken = result.get("time")
# #     image_width = result.get("image", {}).get("width")
# #     image_height = result.get("image", {}).get("height")
# #     predictions = result.get("predictions", [])
# #     animal_names = [prediction.get('class') for prediction in predictions]

# #     # Display each animal name
# #     for i, animal_name in enumerate(animal_names, start=1):
# #         st.write(f"Animal {i}: {animal_name}")
# #     for i in range(5):
# #       if st.checkbox(f'', key=f'uniqueKey{i+4}'):
# #         st.write(f'Option {i} selected')

# #     # If you need to use the animal_names list for further processing, it's now available
# #     # For demonstration, let's print it

# #     st.write("All detected animals:", ", ".join(animal_names))


# #     # Format the result into a string
# #     result_text = f"Inference Time: {time_taken} seconds\nImage Width: {image_width}, Image Height: {image_height}\n"
# #     for i, prediction in enumerate(predictions, start=1):
# #         result_text += (f"\nPrediction {i}:\n"
# #                         f"Class: {prediction.get('class')}\n \n"
# #                         f"Confidence: {prediction.get('confidence')}\n"
# #                         f"Position: (x: {prediction.get('x')}, y: {prediction.get('y')})\n"
# #                         f"Size: (width: {prediction.get('width')}, height: {prediction.get('height')})\n")

# #     # Display the formatted string
# #     st.write(result_text)
# #     return animal_names

# # # if __name__ == "__main__":
# # #     main()

# # def roboflow():
# #     st.title("Inference Form")

# #     st.write("Enter the model details and select the upload method:")

# #     model = st.text_input("Model")
# #     version = st.number_input("Version", step=1)
# #     api_key = st.text_input("API Key")

# #     upload_method = st.radio("Upload Method", ("Upload", "URL"))

# #     url = ""

# #     if upload_method == "Upload":
# #         file = st.file_uploader("Upload File")
# #     else:
# #         url = st.text_input("Enter Image URL", "")

# #     filter_classes = st.text_input("Filter Classes", "")
# #     min_confidence = st.slider("Min Confidence (%)", min_value=0, max_value=100, value=50)
# #     max_overlap = st.slider("Max Overlap (%)", min_value=0, max_value=100, value=50)

# #     inference_result = st.radio("Inference Result", ("Image", "JSON"))

# #     if inference_result == "Image":
# #         show_labels = st.radio("Labels", ("Off", "On"))
# #         stroke_width = st.radio("Stroke Width", ("1px", "2px", "5px", "10px"))

# #     if st.button("Run Inference"):
# #         st.write("Running inference...")

# #         result = perform_inference(model, version, api_key, upload_method, file, url, filter_classes, min_confidence, max_overlap, inference_result)

# #         if result is not None:
# #             if inference_result == "Image":
# #                 st.image(result, caption='Inference Result', use_column_width=True)
# #             else:
# #                 st.json(result)

                
# # # Load the dataset
# # animal_data = pd.read_csv('data/AnimalDataset.csv')

# # def fetch_animal_info_from_dataset(animal_name):
# #     """
# #     Fetch animal information from the provided dataset for names up to three words long.
# #     """
# #     animal_name = animal_name.lower().strip().split()
# #     if len(animal_name) == 1:
# #         info = animal_data[animal_data['Animal'].str.lower() == animal_name[0]].squeeze()
# #     else:
# #         search_query = " ".join(animal_name)  # Re-join for exact match in case of multi-word names
# #         info = animal_data[animal_data['Animal'].str.lower() == search_query].squeeze()
# #     return info if not info.empty else None

# # def display_conservation_status(status):
# #     """
# #     Display conservation status with color coding.
# #     """
# #     status_colors = {
# #         'Least Concern': 'green',
# #         'Vulnerable': 'orange',
# #         'Endangered': 'red',
# #         'Critically Endangered': 'darkred'
# #     }
# #     color = status_colors.get(status, 'blue')
# #     st.markdown(f"<div style='color: {color}; font-size: 24px;'>{status}</div>", unsafe_allow_html=True)

# # def animal_information():
# #     if st.checkbox('Show the entire dataset',key='2'):
# #      st.dataframe(animal_data)

# #     """
# #     Display the animal information from the dataset with enhanced visual presentation.
# #     """

# #     st.subheader("Enter the name of the animal:")
# #     # Add a unique key parameter to the text_input widget
    
# #     # user_input = st.text_input("", "", key="animal_search_input").strip().lower()
# #     user_input=display_text_result(result)
# #     if user_input:
# #         # Filter dataset for matches
# #         matches = animal_data[animal_data['Animal'].str.lower().str.contains(user_input)]['Animal'].unique()
        
# #         if len(matches) > 0:
# #             # If there are matches, show them in a dropdown for the user to select
# #           selected_animal = st.selectbox("Select an animal:", matches, key="animal_select_box")
# #           info = fetch_animal_info_from_dataset(selected_animal)
            
# #           if info is not None:
# #             # Display the animal information
# #             # image_url = info.get('Image_URL', 'https://placekitten.com/200/200')  # Placeholder image
# #             # st.image(image_url, use_column_width=True)
            
# #             # Convert numerical values to string before concatenating
# #             st.markdown("**Height:** " + str(info['Height (cm)']))
# #             st.markdown("**Weight:** " + str(info['Weight (kg)']))
# #             st.markdown("**Color:** " + info['Color'])
# #             st.markdown("**Lifespan:** " + str(info['Lifespan (years)']))
            
# #             st.write("Diet:", info['Diet'])
# #             st.write("Habitat:", info['Habitat'])
# #             st.write("Predators:", info['Predators'])
# #             st.write("Average Speed (km/h):", str(info['Average Speed (km/h)']))
# #             st.write("Countries Found:", info['Countries Found'])
            
# #             st.write("Conservation Status:")
# #             display_conservation_status(info['Conservation Status'])
            
# #             st.write("Family:", info['Family'])
# #             st.write("Gestation Period (days):", str(info['Gestation Period (days)']))
# #             st.write("Top Speed (km/h):", str(info['Top Speed (km/h)']))
# #             st.write("Social Structure:", info['Social Structure'])
# #             st.write("Offspring per Birth:", str(info['Offspring per Birth']))
# #           else:
# #             st.write("No information found for this animal.")
        
            
# # # Add a checkbox to toggle the display of the dataset

# # # Call the main function to display the UI
# # animal_information()





# import streamlit as st
# import pandas as pd
# import base64
# from inference_sdk import InferenceHTTPClient

# # Function Definitions
# def display_text_result(result):
#     if result:
#         # Parse the JSON result
#         time_taken = result.get("time")
#         image_width = result.get("image", {}).get("width")
#         image_height = result.get("image", {}).get("height")
#         predictions = result.get("predictions", [])
#         animal_names = [prediction.get('class') for prediction in predictions]

#         # Display each animal name and inference details
#         st.write("Inference result:")
#         result_text = f"Inference Time: {time_taken} seconds\nImage Width: {image_width}, Image Height: {image_height}\n"
#         for i, animal_name in enumerate(animal_names, start=1):
#             st.write(f"Animal {i}: {animal_name}")
#             prediction = predictions[i-1]
#             result_text += (f"\nPrediction {i}:\n"
#                             f"Class: {animal_name}\n \n"
#                             f"Confidence: {prediction.get('confidence')}\n"
#                             f"Position: (x: {prediction.get('x')}, y: {prediction.get('y')})\n"
#                             f"Size: (width: {prediction.get('width')}, height: {prediction.get('height')})\n")

#         for i in range(5):
#             if st.checkbox(f'Checkbox {i}', key=f'uniqueCheckbox{i}'):
#                 st.write(f'Checkbox {i} is checked.')

#         st.write(result_text)
#         return animal_names
#     return []

# def fetch_animal_info_from_dataset(animal_name, animal_data):
#     animal_name = animal_name.lower().strip().split()
#     if len(animal_name) == 1:
#         info = animal_data[animal_data['Animal'].str.lower() == animal_name[0]].squeeze()
#     else:
#         search_query = " ".join(animal_name)
#         info = animal_data[animal_data['Animal'].str.lower() == search_query].squeeze()
#     return info if not info.empty else None

# def display_conservation_status(status):
#     status_colors = {
#         'Least Concern': 'green',
#         'Vulnerable': 'orange',
#         'Endangered': 'red',
#         'Critically Endangered': 'darkred'
#     }
#     color = status_colors.get(status, 'blue')
#     st.markdown(f"<div style='color: {color}; font-size: 24px;'>{status}</div>", unsafe_allow_html=True)

# def animal_information(result, animal_data):
#     if st.checkbox('Show the entire dataset', key='uniqueKey2'):
#         st.dataframe(animal_data)

#     user_inputs = display_text_result(result)
#     for user_input in user_inputs:
#         # Assuming user_input comes from the predictions directly
#         if user_input:
#             matches = animal_data[animal_data['Animal'].str.lower().str.contains(user_input)].squeeze()
#             if not matches.empty:
#                 info = fetch_animal_info_from_dataset(user_input, animal_data)
#                 if info is not None:
#                     # Display animal information (placeholder for actual content)
#                     st.write(f"Information for {user_input}:")
#                     st.write(info.to_dict())
#                 else:
#                     st.write("No information found for this animal.")

# # Main App Execution
# st.title("Image Inference")

# # Load the dataset
# animal_data = pd.read_csv('data/AnimalDataset.csv')

# # Initialize InferenceHTTPClient
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="coZNWQb0K35yZfbOU3rf"
# )

# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key='fileUploader')
# result = None

# if uploaded_file is not None:
#     st.write("Performing inference...")
#     image_bytes = uploaded_file.read()
#     image_base64 = base64.b64encode(image_bytes).decode("utf-8")
#     result = CLIENT.infer(image_base64, model_id="animal-detection-jvsw5/1")

# animal_information(result, animal_data)


# import streamlit as st
# import pandas as pd

# # Load the dataset
# animal_data = pd.read_csv('data/AnimalDataset.csv')

# def fetch_animal_info_from_dataset(animal_name):
#     """
#     Fetch animal information from the provided dataset for names up to three words long.
#     """
#     animal_name = animal_name.lower().strip().split()
#     if len(animal_name) == 1:
#         info = animal_data[animal_data['Animal'].str.lower() == animal_name[0]].squeeze()
#     else:
#         search_query = " ".join(animal_name)  # Re-join for exact match in case of multi-word names
#         info = animal_data[animal_data['Animal'].str.lower() == search_query].squeeze()
#     return info if not info.empty else None

# def display_conservation_status(status):
#     """
#     Display conservation status with color coding.
#     """
#     status_colors = {
#         'Least Concern': 'green',
#         'Vulnerable': 'orange',
#         'Endangered': 'red',
#         'Critically Endangered': 'darkred'
#     }
#     color = status_colors.get(status, 'blue')
#     st.markdown(f"<div style='color: {color}; font-size: 24px;'>{status}</div>", unsafe_allow_html=True)

# def animal_information():
#     if st.checkbox('Show the entire dataset',key='1'):
#      st.dataframe(animal_data)

#     """
#     Display the animal information from the dataset with enhanced visual presentation.
#     """
#     st.subheader("Enter the name of the animal:")
#     # Add a unique key parameter to the text_input widget
#     user_input = st.text_input("", "", key="animal_search_input").strip().lower()
    
#     if user_input:
#         # Filter dataset for matches
#         matches = animal_data[animal_data['Animal'].str.lower().str.contains(user_input)]['Animal'].unique()
        
#         if len(matches) > 0:
#             # If there are matches, show them in a dropdown for the user to select
#           selected_animal = st.selectbox("Select an animal:", matches, key="animal_select_box")
#           info = fetch_animal_info_from_dataset(selected_animal)
            
#           if info is not None:
#             # Display the animal information
#             # image_url = info.get('Image_URL', 'https://placekitten.com/200/200')  # Placeholder image
#             # st.image(image_url, use_column_width=True)
            
#             # Convert numerical values to string before concatenating
#             st.markdown("**Height:** " + str(info['Height (cm)']))
#             st.markdown("**Weight:** " + str(info['Weight (kg)']))
#             st.markdown("**Color:** " + info['Color'])
#             st.markdown("**Lifespan:** " + str(info['Lifespan (years)']))
            
#             st.write("Diet:", info['Diet'])
#             st.write("Habitat:", info['Habitat'])
#             st.write("Predators:", info['Predators'])
#             st.write("Average Speed (km/h):", str(info['Average Speed (km/h)']))
#             st.write("Countries Found:", info['Countries Found'])
            
#             st.write("Conservation Status:")
#             display_conservation_status(info['Conservation Status'])
            
#             st.write("Family:", info['Family'])
#             st.write("Gestation Period (days):", str(info['Gestation Period (days)']))
#             st.write("Top Speed (km/h):", str(info['Top Speed (km/h)']))
#             st.write("Social Structure:", info['Social Structure'])
#             st.write("Offspring per Birth:", str(info['Offspring per Birth']))
#           else:
#             st.write("No information found for this animal.")
            
# # Add a checkbox to toggle the display of the dataset

# # Call the main function to display the UI
# animal_information()
