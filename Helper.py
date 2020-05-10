import os
from cv2 import imread, rectangle
import matplotlib.pyplot as plt

def img_disp(img_path, bbox, box_width=10):
    # Read image
    img = imread(img_path)
    
    # Get the bounding box
    (x, y, w, h) = bbox
        
    # Add the bounding box
    rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), box_width)

    # Show image with bounding box
    plt.imshow(img)
    plt.show() 

def create_dir(dir_name):
    """
    Create a folder to store augmented images (if it doesn't currently exist).
    Note: assumes dir_name ends with '/' (else remove '[:-1]')
    """
    
    dirs = dir_name.split('/')[:-1] 
    cwd = os.getcwd()+'/'
    
    for i in range(len(dirs)):
        dir_name = '/'.join(dirs[0:i+1])
        if not os.path.exists(cwd+dir_name):
            os.mkdir(cwd+dir_name)

def get_imageName(img_path):
    return img_path.split('/')[2].split('.')[0]
    
def get_iou(box1, box2):
    """
    Compute the Intersection-Over-Union of two given boxes.
    Args:
      box1: array of 4 elements [cx, cy, width, height].
      box2: same as above
    Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
    """

    lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    
    if lr > 0:
        tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    
        if tb > 0:
            intersection = tb*lr
            union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

            return intersection/union

    return 0