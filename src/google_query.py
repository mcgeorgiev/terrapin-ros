from google.cloud import vision
from google.cloud.vision.feature import Feature
from google.cloud.vision.feature import FeatureTypes
import io
import Image
# with io.open("/home/michael/terrapin/objects.jpg", 'rb') as image_file:
#     content = image_file.read()


class GoogleVision:
    def __init__(self):
        self.client = vision.Client()

    def query(self, rgb_image):
        bytes = rgb_image.tobytes()
        img = Image.fromarray(rgb_image)
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        image = self.client.image(content = imgByteArr)
        features = [Feature(FeatureTypes.LABEL_DETECTION, 1), Feature(FeatureTypes.FACE_DETECTION,1)]
        annotations = image.detect(features)


        for thing in annotations:
            for label in thing.labels:
                return label.description, label.score
                break
            break
