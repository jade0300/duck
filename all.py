import cv2
import numpy as np
import imutils
import pytesseract
import base64

pytesseract.pytesseract.tesseract_cmd = r".\Tesseract-OCR\tesseract.exe"

def baisi(img):
    #灰階
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #邊緣檢測
    edged_img = cv2.Canny(gray_img, 30, 500)
    # cv2.imshow(f'gray',gray_img)
    
    #尋找輪廓
    contours=cv2.findContours(edged_img.copy(),cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.08* perimeter, True)
        if len(approx) == 4: #矩形
            plate_contour = approx
            break
    #遮罩
    mask = np.zeros(gray_img.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, (255, 255, 255), -1)
    #and邏輯運算
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    #擷取車牌的部分
    min_x=min(plate_contour[0][0][0],plate_contour[1][0][0],plate_contour[2][0][0],plate_contour[3][0][0])
    min_y=min(plate_contour[0][0][1],plate_contour[1][0][1],plate_contour[2][0][1],plate_contour[3][0][1])
    max_x=max(plate_contour[0][0][0],plate_contour[1][0][0],plate_contour[2][0][0],plate_contour[3][0][0])
    max_y=max(plate_contour[0][0][1],plate_contour[1][0][1],plate_contour[2][0][1],plate_contour[3][0][1])
    cut_img=img[min_y:max_y,min_x:max_x]
    return gray_img,edged_img,masked_image,cut_img

#輸入影像
def open_img(img_t):
    img = cv2.resize(img_t, None, fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    gray_img,edged_img,masked_image,cut_img=baisi(img)
    # cv2.imshow(f'{x}_gray',gray_img)
    # cv2.imshow(f'{x}_edged',edged_img)
    # cv2.imshow(f'{x}_mask', masked_image)
    # cv2.imshow(f'{x}_cut',cut_img)
    cv2.imwrite(f'./img/z_cut.jpg',cut_img)
    return cut_img

#銳化
def sharpen(img, sigma=25):    
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def start(tmp):
    cut_img=open_img(tmp)
    gray_cut = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(f'x_cut_gray',gray_cut)
    gray_cut_sharpen = sharpen(gray_cut)
    # cv2.imshow(f'x_cut_gray_sharpen',gray_cut_sharpen)
    ret, gray_cut_sharpen_threshold = cv2.threshold(gray_cut_sharpen, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow(f'x_cut_gray_sharpen_threshold',gray_cut_sharpen_threshold)
    result = pytesseract.image_to_string(gray_cut_sharpen_threshold, lang="eng")
    result=result.replace('\n','')
    print(result)

    return result

def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str


cv2.waitKey(0) 
cv2.destroyAllWindows()
    