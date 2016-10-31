
import numpy as np
from skimage import draw



# draw on img
def img_with_bbox(img_origin, gt_1, c=1):    
    img =np.copy(img_origin)
    maxh, maxw = img.shape[:2]
    gt_1 = [int(i) for i in gt_1]
    tl_x, tl_y, w, h = gt_1
    
    if tl_x+w >= maxw:
        w = maxw - tl_x -1
    if tl_y+h >= maxh:
        h = maxh - tl_y -1
        
    tr_x, tr_y = tl_x + w, tl_y 
    dl_x, dl_y = tl_x, tl_y + h
    dr_x, dr_y = tl_x + w, tl_y +h

    rr1, cc1 = draw.line( tl_y,tl_x, tr_y, tr_x)
    rr2, cc2 = draw.line( tl_y,tl_x, dl_y, dl_x)
    rr3, cc3 = draw.line( dr_y,dr_x, tr_y, tr_x)
    rr4, cc4 = draw.line( dr_y,dr_x, dl_y, dl_x)
    img[rr1, cc1, :] = c
    img[rr2, cc2, :] = c
    img[rr3, cc3, :] = c
    img[rr4, cc4, :] = c
    return img

