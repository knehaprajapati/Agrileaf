import streamlit as st
import tensorflow as tf
import numpy as np
import json

# Production Optimization: Model loading ko cache kar diya taaki app crash na ho
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model("Trained_model.keras")

# tensorflow model function
def model_prediction(test_image):
    # Caching use ki hai taaki har prediction par model load na ho
    model = load_model_cached()
    # Streamlit file_uploader se mili image ko load karna
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # batch conversion
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)  
    return result_index

# Load disease treatment & prevention data
with open("diseases.json", "r") as f:
    disease_data = json.load(f)

# UI 
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if (app_mode == "Home"):
    st.header("Plant Diseases Recognition System and Provide Treatment + Prevention")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases and provide well-treatment and prevention information. Together, let's protect our crops and ensure a healthier harvest!
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases and provide the treatment and prevention of that leaf.
    3. **Results:** View the results and recommendations for further action.
    4. **Leaves which you can upload:** Apple, Blueberry, Cherry, Corn(maize), Grape, Orange, Peach, Pepper(bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection with 97 percent accuracy.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
                """)

# About page
elif(app_mode == "About"):
    st.header("About")
    st.markdown("""
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    ###Content
    1. Train (70295 images)
    2. Test (33 images)
    3. Valid (17527 images)
    4. Diseases json for treatment and prevention information
    """)
    
# Prediction Page
elif(app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'png'])
    
    if test_image is not None:
        if (st.button("Show Image")):
            st.image(test_image, width='stretch')
        
        # predict button
        if (st.button("Predict")):
            with st.spinner("Please Wait..."):
                result_index = model_prediction(test_image)
                
                # Define Classes
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                             'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                             'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                             'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                             'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                             'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                             'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                             'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                             'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                             'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                             'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                             'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                             'Tomato___healthy']
                
                st.success("Model is Predicting it's a **{}**".format(class_name[result_index]))
                predicted_disease = class_name[result_index]

                # Fetch treatment & prevention info
                info = disease_data.get(predicted_disease, None)

                if info:
                    st.subheader("🩺 Treatment")
                    for t in info["treatment"]:
                        st.write("✔️", t)

                    st.subheader("🛡️ Prevention")
                    for p in info["prevention"]:
                        st.write("✔️", p)

                    st.subheader("📚 More Information")
                    for l in info["link"]:
                        st.markdown(f"🔗 [Read here]({l})")
                else:
                    st.warning("Treatment & prevention information not available for this disease.")