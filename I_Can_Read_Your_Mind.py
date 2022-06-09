
import cv2
import numpy as np

cap = cv2.VideoCapture("magic_draw2.mp4")
# cap = cv2.VideoCapture(0)
 
def calmagnitude(img):
    img = np.float32(img) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1) 
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # print("Gradient direction:\n",angle)
    # print("Gradient magnitude:\n",mag)
    
    return mag,angle

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
_,first_img = cap.read()

first_img = cv2.resize(first_img,(600,400))    
    
first_gray = cv2.cvtColor(first_img,cv2.COLOR_BGR2GRAY)

first_gray[164:220,170:190]=0


p0 = cv2.goodFeaturesToTrack(first_gray, mask = None, **feature_params)





mask = np.zeros_like(first_img)

mask2 = np.zeros_like(first_img)


frame_count =0

while True:
    
    
    
    _,img = cap.read()

    img = cv2.resize(img,(600,400))    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray,gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    if (frame_count<=70 and frame_count>=0):
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            img = cv2.circle(img,(int(a),int(b)),5,color[i].tolist(),-1)
        img = cv2.add(img,mask)
        
    if (frame_count<=250 and frame_count>=160):
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask2 = cv2.line(mask2, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            img = cv2.circle(img,(int(a),int(b)),5,color[i].tolist(),-1)
        img = cv2.add(img,mask2)
          
    
    # gray = gray[164:220,170:190]
    
    frame_count +=1
   
  
    
    # mang,ang = calmagnitude(gray)
    
    
    print(frame_count)
    # print(mang)
    # print(ang)
    
    cv2.imshow("image",np.hstack([img,mask,mask2]))

    if cv2.waitKey(30) & 0xFF == ord("q") or frame_count==877:
        
       break
    first_gray = gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
# cv2.imshow("shape1",mask)
# cv2.waitKey(0) 