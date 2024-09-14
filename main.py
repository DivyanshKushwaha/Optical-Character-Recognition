import pandas as pd
from models.model import cnn_model
from src.utils import download_images
from notebooks.image_preprocessing import preprocess_image

def main():
    train_data = pd.read_csv('dataset/train.csv')
    image_links = train_data['image_link'].tolist()

    download_images(image_links)

    for idx in range(len(image_links)):
        image_path = f'images/image_{idx}.jpg'
        resized_image, text= preprocess_image(image_path)

    model = cnn_model()
    model.save('models/trained_model.h5')

if __name__=="__main__":
    main()
