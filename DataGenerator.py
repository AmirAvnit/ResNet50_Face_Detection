import keras
import cv2
import numpy as np
from Helper import get_imageName, get_iou, create_dir

class RandCropBatchGen(keras.utils.Sequence):
    
    def __init__(self,crop_aug, flip_aug, face_cascade, face_files, bbox_dict, set_name, batch_size=128, batch_nb=10, 
                  IoU_thresh=(0.2,0.5), start_ind=0, pos_n=32, neg_n=96, resize_dim=None, shuffle=True,
                  dim=(224,224),channels_nb=3, save_files=False):
        
        'Initialization'
        self.crop_aug = crop_aug
        self.flip_aug = flip_aug
        self.face_cascade = face_cascade
        
        self.face_files = face_files     
        self.bbox_dict = bbox_dict      
        self.set_name = set_name
        self.batch_size = batch_size
        self.batch_nb = batch_nb
        self.IoU_thresh = IoU_thresh
        self.start_ind = start_ind
        self.end_ind = start_ind + batch_size
        self.inds = np.arange(self.start_ind,self.end_ind)
        self.pos_n = pos_n
        self.neg_n = neg_n
        self.resize_dim = resize_dim
        self.shuffle = shuffle
        self.batch_nb = 1
        self.save_files = True
        
        self.dim = dim
        self.channels_nb = channels_nb
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batch_nb
    
    def __getitem__(self,ind):
        'Generate one batch of data'
        X,y = self.__data_generation()
        return X,y
    
    def on_epoch_end(self):
        # Update indices for next iteration
        self.start_ind = self.end_ind
        self.end_ind = self.start_ind + self.batch_size
        
        if self.end_ind >= len(self.face_files)-1: # If exceeding set size, stop at last image  
            self.end_ind = len(self.face_files)-1
        
        if self.end_ind - self.start_ind < 2:       # If both indices are set to last image, start from the first
            self.start_ind = 0
            self.end_ind = self.batch_size
        
        self.inds = np.arange(self.start_ind,self.end_ind)
        
        # shuffle indices order for each batch
        if self.shuffle:
            np.random.shuffle(self.inds)
                                          
    
    def __data_generation(self):
        
        # Initialization
        pos_count = 0
        neg_count = 0
        X = np.empty((self.batch_size, *self.dim, self.channels_nb))
#         y = np.empty((self.batch_size), dtype=int)
        y = np.empty((self.batch_size,2), dtype=int)
        imgs_names = []
        
        """
        For efficieny, begin with creating an equal number of random crops to batch size.
        We later complete missing positive/negative samples to create a full batch.
        """
        
        missing = self.pos_n + self.neg_n # Number of missing samples for full batch

        while missing > 0: # Iterate while there are missing samples
            
            # Get randomly ordered list of file paths
            face_paths = [self.face_files[ind] for ind in self.inds]
            
            # Get image arrays and names
            imgs = [cv2.imread(path) for path in face_paths]

            # and their corresponding face bounding boxes 
            imgs_bboxes = [self.bbox_dict[path] for path in face_paths]

            # Create randomly cropped images and get their coordinates
            cropped,aug_coords = self.crop_aug.augment_images(imgs)

            # Get IoUs of random crops and face bounding boxes
            IoUs = [get_iou(aug_coords[i],imgs_bboxes[i]) for i in range(len(cropped))]
            
            if self.save_files:
                save_dir = 'AugImgs/' + self.set_name + '/batch_' + str(self.batch_nb) + '/'
                create_dir(save_dir)
            
            for i in range(len(cropped)):
                
                if(IoUs[i] >= self.IoU_thresh[1]):
                    if pos_count < self.pos_n:
                        flipped = self.flip_aug.augment_image(cropped[i])        # Random horizontal flip
                        if self.resize_dim is not None:
                            flipped = cv2.resize(flipped,self.resize_dim)        # Resize to network input dimensions
                        if self.save_files:                                      # Save image file
                            cv2.imwrite(save_dir + get_imageName(face_paths[i]) 
                                    + '_' + str(1) + '.jpg',flipped)   
                        
                        flipped = flipped.astype('float32')/255                  # Normalize to [0,1]
                        flipped = np.asarray(flipped)
                        X[pos_count+neg_count] = flipped
#                         y[pos_count+neg_count] = 1
                        y[pos_count+neg_count] = [1,0]
                        pos_count += 1

                elif(IoUs[i] <= self.IoU_thresh[0]):
                    if neg_count < self.neg_n:
                        
                        # Use pretrained detector to additionaly check for the presence of faces 
                        faces = self.face_cascade.detectMultiScale(cropped[i])

                        if len(faces)==0: # Save as non-face if no faces were detected
                            flipped = self.flip_aug.augment_image(cropped[i])        # Random horizontal flip
                            if self.resize_dim is not None:
                                flipped = cv2.resize(flipped,self.resize_dim)        # Resize to network input dimensions
                            if self.save_files:                                      # Save image file
                                cv2.imwrite(save_dir + get_imageName(face_paths[i]) 
                                    + '_' + str(0) + '.jpg',flipped)   
                            
                            flipped = flipped.astype('float32')/255                  # Normalize to [0,1]
                            flipped = np.asarray(flipped)                 
                            X[pos_count+neg_count] = flipped
#                             y[pos_count+neg_count] = 0
                            y[pos_count+neg_count] = [0,1]
                            neg_count += 1

            # Update number of missing samples to create a full batch
            missing = self.pos_n + self.neg_n - pos_count - neg_count     

            # if samples are still missing, update start and end indices
            # Note: I avoid running augmenter on a single sample, which would result in an error
            if missing > 0:
                self.start_ind = self.end_ind
                self.end_ind = self.start_ind + missing + 1
                
                if self.end_ind >= len(self.face_files)-1: # If exceeding set size, stop at last image  
                    self.end_ind = len(self.face_files)-1
                    
                if self.end_ind - self.start_ind < 2:
                    self.start_ind = 0
                    self.end_ind = self.start_ind + missing + 1
                    
                self.inds = np.arange(self.start_ind,self.end_ind)
                
                # shuffle indices order 
                if self.shuffle:
                    np.random.shuffle(self.inds)
        
        print('{} batch #{} ready'.format(self.set_name,self.batch_nb))
        self.batch_nb+=1
            
        return X,y  