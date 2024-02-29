from django.shortcuts import render,redirect

import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

from complaint.models import Complaint

# Create your views here.

def complaint(request):
    if (request.user.is_authenticated):
        return render(request,'complaint/complaint.html')
    else:
        return render(request,'user/signin.html')

def add_complaint(request):
    if request.method == "POST":
        user = request.user
        violation_date=request.POST.get("vdate")
        place = request.POST.get("place")
        district=request.POST.get("district")
        state=request.POST.get("state")
        comment=request.POST.get("comment")
        uploaded_file = request.FILES.get('evidence')
        complaint = Complaint(
                    user=user,
                    violation_date=violation_date,
                    place=place,
                    district=district,
                    state=state,
                    comment=comment,
                    # Set other fields similarly
                    uploaded_file=uploaded_file,
                )

        # Save the model to the database
        complaint.save()

        verify(request, complaint.id)
        content={
            "success":1
        }
        return render(request,'pages/index.html',content)

# views.py
# from django.shortcuts import render, redirect
# from .forms import ComplaintForm

# def create_complaint(request):
#     if request.method == 'POST':
#         form = ComplaintForm(request.POST, request.FILES)
#         if form.is_valid():
#             complaint = form.save(commit=False)

#             # Save file to the model instance
#             complaint.uploaded_file = form.cleaned_data['uploaded_file']

#             # Save the model to the database
#             complaint.save()

#             # Do any additional processing or redirect as needed
#             return redirect('success_page')
#     else:
#         form = ComplaintForm()

#     return render(request, 'complaint/complaint.html', {'form': form})

def complaints(request):
    complaints=Complaint.objects.filter(user=request.user).order_by('-id')
    content={
        "complaints":complaints
    }
    return render(request,'complaint/complaints.html',content)

def cancel(request,id):
    Complaint.objects.filter(id=id).delete()
    content={
            "delete":1
        }
    return render(request,'pages/index.html',content)


# ML part


def verify(request, cid):
    np.random.seed(20)

    classesFilePath = "__background__ person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant street sign stop sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe hat backpack umbrella shoe eye glasses handbag tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball glove skateboard surfboard tennis racket bottle plate wine glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch potted plant bed mirror dining table window desk toilet door tv laptop mouse remote keyboard cell phone microwave oven toaster sink refrigerator blender book clock vase scissors teddy bear hair drier toothbrush hair brush"
    classesList = classesFilePath.split()

    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
    # modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz"
    # imagePath = "../../Data/helmet/images/BikesHelmets56.png"



    # with open(classesFilePath, "r") as f:
    #     classesList = f.read().splitlines()

    colorList = np.random.uniform(low=0, high=255, size=(len(classesList), 3))
    print(len(classesList), len(colorList))



    fileName = os.path.basename(modelURL)
    modelName = fileName[:fileName.index('.')]
    print("fileName: ", fileName, "ModelName: ", modelName)

    cacheDir = "./pretrained_models"
    # Creating the models directory if not existed before
    os.makedirs(cacheDir, exist_ok=True)
    get_file(fileName, origin=modelURL, cache_dir=cacheDir, cache_subdir="models", extract=True)



    model1 = tf.saved_model.load(os.path.join(cacheDir, "models", modelName, "saved_model"))


    complaint=Complaint.objects.get(id=cid)
    print(complaint)
    print(complaint.uploaded_file.path)
    vdo = cv2.VideoCapture(complaint.uploaded_file.path)
    print(vdo)
    print(vdo.isOpened()) 
    detected = 0

    t1 = 0
    while True:
        captured, frame = vdo.read()
        # print(frame)
        if not captured:
                break  # End of video
        
        print(frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imageRGB = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        imageTensor = tf.convert_to_tensor(imageRGB, dtype=tf.uint8)
        imageTensor = imageTensor[tf.newaxis, ...]

        detections = model1(imageTensor)

        bboxes = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape
        # image = cv2.resize(image, (1000, int(imH * (1000/imW))))
        # imH, imW = (int(imH * (1000/imW)), 1000)

        # gives indexes of bboxes with the criteria
        bboxIds = tf.image.non_max_suppression(bboxes, classScores, max_output_size=50, iou_threshold=0.6, score_threshold=0.3)


        if len(bboxes):
            for i in bboxIds:
                bbox = tuple(bboxes[i])
                classIndex = classIndexes[i]
                classLabel = classesList[classIndex]
                classConfidence = round(100*classScores[i])
                classColor = colorList[classIndex]

                displayTxt = f'{classLabel}: {classConfidence}%'
                # displayTxt = '{}: {}%'.format(classLabel, classConfidence)
                ymin, xmin, ymax, xmax = bbox   # in relative format
                ymin, xmin, ymax, xmax = (int(ymin*imH), int(xmin*imW), int(ymax*imH), int(xmax*imW))

                if classLabel in ["motorcycle"]:
                    detected = 1
                    break
                    # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                    # cv2.putText(frame, displayTxt, (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX_SMALL,  1, classColor, 1)

                # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                # cv2.putText(frame, displayTxt, (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX_SMALL,  1, classColor, 1)


        t2 = time.time()
        fps = 1/(t2 - t1)
        t1 = t2

        # cv2.putText(frame, f"FPS: {int(fps)}", (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 200, 0), 1)
        # cv2.imshow("Video", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    vdo.release()
    cv2.destroyAllWindows()
    
    print(detected)
    if detected:
        complaint.status='Processed'
        complaint.save()
    else:
        complaint.status='Rejected'
        complaint.save()
    print(complaint)