from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "peopleCrossing.png"), output_image_path=os.path.join(execution_path, "imageResult.jpg"), minimum_percentage_probability=50)
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "COCOImages/magic9.jpeg"), output_image_path=os.path.join(execution_path, "imageResult.jpg"),  minimum_percentage_probability=75)
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
    print("--------------------------------")
'''
directory = os.fsencode('COCOImages')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    theFile = 'COCOImages/' + filename
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, theFile), output_image_path=os.path.join(execution_path, "imageResult.jpg"),  minimum_percentage_probability=75)
    for eachObject in detections:
        print(filename)
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        print("--------------------------------")
'''