from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("SymDescribeImages/models/detection_model-ex-009--loss-0025.046.h5")
detector.setJsonPath("SymDescribeImages/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="SymDescribeImages/train/images/IMG_0013_jpg.rf.87aaa7b7261cc978832be920b79df06a.jpg", output_image_path="detectedResult.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])