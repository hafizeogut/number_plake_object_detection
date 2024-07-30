from ultralytics import YOLO
from easyocr import Reader
import time
import torch
import cv2
import os 
import csv
CONFIDENCE_THRESHOLD = 0.4
COLOR = (0, 255, 0)

def detect_number_plates(image, model, display=False):
    start = time.time()
    # pass the image through the model and get the detections :# görüntüyü modelden geçirin ve tespitleri alın
    detections = model.predict(image)[0].boxes.data
    print("detections",detections) #/home/hafizeogut/Desktop/number_plake_object_detection/datasets/images/train/cc701755-33.jpeg
    # check to see if the detections tensor is not empty:# algılama tensörünün boş olup olmadığını kontrol edin
    if detections.shape != torch.Size([0, 6]):

        # initialize the list of bounding boxes and confidences :# sınırlayıcı kutuların ve gizliliklerin listesini başlat
        boxes = []
        confidences = []

        # loop over the detections:# algılamalar üzerinde döngü:
        for detection in detections:
            # extract the confidence (i.e., probability) associated:# ilişkili güveni (yani olasılığı) çıkarın:
            # with the prediction:# tahmin ile
            confidence = detection[4]#0.9205  #etections tensor([[128.2224, 212.9803, 282.7975, 257.0309,   0.9205,   0.0000]], device='cuda:0')   

            # filter out weak detections by ensuring the confidence:# güveni sağlayarak zayıf tespitleri filtreleyin
            # is greater than the minimum confidence: minimum güvenden daha büyük ise
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence, add:# eğer güven minimum güvenden büyükse ekle
            # the bounding box and the confidence to their respective lists:# sınırlayıcı kutu ve ilgili listeye olan güven
            boxes.append(detection[:4]) #128.2224, 212.9803, 282.7975, 257.0309,
            confidences.append(detection[4]) #0.9205

        print(f"{len(boxes)} Number plate(s) have been detected.")
        # initialize a list to store the bounding boxes of the:# sınırlayıcı kutuları saklamak için bir liste başlat
        # number plates and later the text detected from them :# plaka ve daha sonra plakalardan tespit edilen metin
        number_plate_list= []

        # loop over the bounding boxes:# sınırlayıcı kutuların üzerinde döngü
        for i in range(len(boxes)):
            # extract the bounding box coordinates:# sınırlayıcı kutu koordinatlarını çıkar:
            xmin, ymin, xmax, ymax = int(boxes[i][0]), int(boxes[i][1]),\
                                     int(boxes[i][2]), int(boxes[i][3])
            # append the bounding box of the number plate:# plakanın sınırlayıcı kutusunu ekleyin
            number_plate_list.append([[xmin, ymin, xmax, ymax]])

            # draw the bounding box and the label on the image :# görselin üzerine sınırlayıcı kutuyu ve etiketi çizin
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "Number Plate: {:.2f}%".format(confidences[i] * 100)
            cv2.putText(image, text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

            if display:
                # crop the detected number plate region:# tespit edilen plaka bölgesini kırpın
                number_plate = image[ymin:ymax, xmin:xmax]
                # display the number plate:# plakayı göster
                cv2.imshow(f"Number plate {i}", number_plate)

        end = time.time()
        # show the time it took to detect the number plates:# plakaları tespit etmek için geçen süreyi göster
        print(f"Time to detect the number plates: {(end - start) * 1000:.0f} milliseconds")
        # return the list containing the bounding:# sınırlamayı içeren listeyi döndür:
        # boxes of the number plates
        return number_plate_list
    # if there are no detections, show a custom message:# herhangi bir algılama yoksa özel bir mesaj göster
    else:
        print("No number plates have been detected.")
        return []

def recognize_number_plate(image_or_path,reader,number_plate_list,write_to_csv=False):
                                        #OCR gerçekleştirmek için kullanacağımız,elde edilen plaka listesi, 
    start = time.time()
    # if the image is a path, load the image; otherwise, use the image:# eğer resim bir yol ise resmi yükleyin; aksi takdirde görseli kullanın
    image = cv2.imread(image_or_path) if isinstance(image_or_path,str) else image_or_path
    print("number_plate_list",number_plate_list)
    for i,box in enumerate(number_plate_list):
        #crop the number plate region:#plaka bölgesini kırp
        np_image = image[box[0][1]:box[0][3], box[0][0]:box[0][2]]

        # detect the text from the license plate using the EasyOCR reader :# EasyOCR okuyucusunu kullanarak plaka üzerindeki metni tespit edin
        detection = reader.readtext(np_image,paragraph = False)
                                            # Algılanan metni otomatik olarak birleştirmesini söyler.
#                                             Time to detect the number plates: 825 milliseconds
# [([[8, 14], [170, 14], [170, 46], [8, 46]], 'HR 26CT4063', 0.8096237572714008)]-> False
# [[[[8, 14], [170, 14], [170, 46], [8, 46]], 'HR 26CT4063']]->True

        print("/n recognize_number_plate detection",detection)
        if len(detection)  == 0:
            text = ""
        else:
            text = str(detection[0][1])
        number_plate_list[i].append(text)


        #print("number_plate_list",number_plate_list)

        #print(detection)#[([[5, 11], [113, 11], [113, 49], [5, 49]], 'SKRISTY', 0.594537148199287)]
                            #Sınırlayıcı kutular:                        #Güven Puanı            s 

    if write_to_csv:
        with open ("number_plates.csv","w",newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["image_path","box","text"])

            for box,text in number_plate_list:
                csv_writer.writerow([image_or_path,box,text])

        csv_file.close()

    end = time.time()
    print(f"Time to recognize the number plates: {(end-start)*1000:.0f}")
    return number_plate_list

# if this script is executed directly, run the following code:# if this script is executed directly, run the following code
if __name__ == "__main__":

    # load the model from the local directory:# modeli yerel dizinden yükleyin
    model = YOLO("/home/hafizeogut/Desktop/number_plake_object_detection/runs/detect/number_plake_train/weights/best.pt")
    # initialize the EasyOCR reader:# EasyOCR okuyucuyu başlat
    reader = Reader(['en'], gpu=True)

    # path to an image or a video file :# bir görselin veya video dosyasının yolu
    file_path = "/home/hafizeogut/Desktop/number_plake_object_detection/datasets/images/test/e44c62aa-194.jpeg"
    # Extract the file name and the file extension from the file path:# Dosya adını ve dosya uzantısını dosya yolundan çıkarın:
    _, file_extension = os.path.splitext(file_path)

    # Check the file extension:# Dosya uzantısını kontrol edin
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print("Processing the image...")
        reader = Reader(["en"],gpu = True)
        image = cv2.imread(file_path)
        number_plate_list = detect_number_plates(image, model,
                                                 display=True)
        
        # recognize_number_plates(file_path,reader,number_plate_list)
        cv2.imshow('Image', image)
        cv2.waitKey(1) 
 
        print("main number plate list",number_plate_list)
        if number_plate_list != []:
            number_plate_list = recognize_number_plate(file_path,reader,number_plate_list,write_to_csv=True)

            for box,text in number_plate_list:
                cv2.putText(image,text,(box[0],box[3]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR,2)

            cv2.imshow("Image",image)
            cv2.waitKey(0)   
    elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
        print("Processing the video...")

        video_cap = cv2.VideoCapture(file_path)

        # grab the width and the height of the video stream:# video akışının genişliğini ve yüksekliğini yakalayın
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        # initialize the FourCC and a video writer object:# FourCC'yi ve video yazıcı nesnesini başlat
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, fps,
                                 (frame_width, frame_height))

        # loop over the frames:# çerçeveler üzerinde döngü
        while True:
            # starter time to computer the fps:# fps'yi bilgisayara aktarma zamanı
            start = time.time()
            success, frame = video_cap.read()

            # if there is no more frame to show, break the loop:# if there is no more frame to show, break the loop
            if not success:
                print("There are no more frames to process."
                      " Exiting the script...")
                break

            number_plate_list = detect_number_plates(frame, model)
            if number_plate_list != []:
                number_plate_list = recognize_number_plate(frame,reader,number_plate_list,write_to_csv=True)

                for box,text in number_plate_list:
                    cv2.putText(frame,text,(box[0],box[3]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR,2)
             

            # end time to compute the fps:# fps'yi hesaplamak için bitiş zamanı
            end = time.time()
            # calculate the frame per second and draw it on the frame:# saniyedeki kareyi hesaplayın ve karenin üzerine çizin
            fps = f"FPS: {1 / (end - start):.2f}"
            cv2.putText(frame, fps, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

            # show the output frame:# çıktı çerçevesini göster
            cv2.imshow("Output", frame)
            # write the frame to disk:# çerçeveyi diske yaz
            writer.write(frame)
            # if the 'q' key is pressed, break the loop:# 'q' tuşuna basılırsa döngüyü kır
            if cv2.waitKey(10) == ord("q"):
                break

        # release the video capture, video writer, and close all windows
        video_cap.release()
        writer.release()
        cv2.destroyAllWindows()
