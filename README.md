# arts_classifier
 classifier oImagef different styles of arts: drawings, painting, sculpture, engraving, old Russian art
 Max accuracy: 86%


<b>Dataset</b> is uploaded [here](https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving)



1) Install dependencies

        pip install tensorflow opencv2-python numpy

2) Generate csv files with image paths and correspoding labels, runnig generate_csv.py

        python generate_csv.py

3) Create training.tfrecords and validation.tfrecords files for tensorflow Dataset API via

        python create_dataset.py
    
4) Train and evaluate using

        python train.py
    
5) Use "predict-playground.py" to make predictions for test set or for your own image

        python predict-playground.py
        python predict-playground.py --image path/to/image.jpg
    
