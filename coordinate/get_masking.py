import cv2

COLOR = (0, 255, 0)
CAM_NUM = 0     # defaults : 0
WIDTH = 640
HEIGHT = 480

FILE_DIR = 'coordinate'
FILE_NAME = 'mask_cordi.txt'    # must include '.txt'

def click_event(event, x, y, flags, params):
    
    if event == cv2.EVENT_LBUTTONDOWN:

        cv2.circle(frame, (x,y), 2, COLOR, -1)
        cv2.imshow('frame', frame)

        save_file = FILE_DIR+'/'+FILE_NAME if  FILE_DIR != '' else FILE_NAME

        with open(save_file, 'a') as f:
            f.write(f'{x} {y} ')

if __name__=="__main__":

    cap = cv2.VideoCapture(CAM_NUM)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    ref, frame = cap.read()
    assert ref

    cap.release()

    cv2.imshow('frame', frame)

    cv2.setMouseCallback('frame', click_event)

    cv2.waitKey(0)

    cv2.destroyAllWindows()