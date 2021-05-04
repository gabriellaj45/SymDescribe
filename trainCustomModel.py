from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="SymDescribeImages")
trainer.setTrainConfig(object_names_array=["three", "skull", "water", "fire", "normal", "spade", "electric", "diamond",
                                           "pause", "heart", "club", "dark", "psychic", "joker", "one",
                                           "plus", "dots", "grid", "circle", "tree skull", "fighting", "squiggly",
                                           "grass", "tree", "asterisk", "two", "seven"],
                       batch_size=32, num_experiments=50, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()