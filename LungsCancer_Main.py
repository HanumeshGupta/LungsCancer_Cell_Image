import os 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Calling Dataset

df_train = pd.read_csv(r"D:\Visual Studio Code\ML\LungCancerImg\lung_cancer_train.csv")

df_train = df_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Calling Encoder From the Database:

encoder = LabelEncoder()

df_train["label"] = encoder.fit_transform(
    df_train["lung_cancer_type"]
)

NUM_CLASSES = len(
    encoder.classes_
)

print("Encoder : ",encoder.classes_)

#Calling Saved Model:
Address_Model = r"D:\Visual Studio Code\ML\lung_cancer_classifier.keras"
model = load_model(Address_Model)
print("Model loaded successfully")




# Loading Image in 224 pixel
IMG_SIZE = (224,224)

def load_image(path, label):

    img = tf.io.read_file(path)

    img = tf.image.decode_image(
        img,
        channels=3,
        expand_animations=False
    )

    img = tf.image.resize(
        img,
        IMG_SIZE
    )

    img = preprocess_input(img)

    return img, label



# Predict Img Function
def predict_image(img_path):

    # Loading Image 
    img = image.load_img(
        img_path,
        target_size=(224,224)
    )

    
    img_array = image.img_to_array(
        img
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    prediction = model.predict(
        img_array
    )

    predicted_class = np.argmax(
        prediction
    )

    label = encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = np.max(
        prediction
    )

    print(
        f"Prediction: {label}"
    )

    print(
        f"Confidence: {confidence:.2%}"
    )

    return label

# Path of Image
img_path = r"D:\Visual Studio Code\ML\LungCancerImg\Data\valid\adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib\000108 (7).png"

label = predict_image(img_path)

print("Prediction:", label)
