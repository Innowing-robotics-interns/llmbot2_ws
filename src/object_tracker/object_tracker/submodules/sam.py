from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

class SAM:
    def __init__(self):
        sam_checkpoint = "/home/fyp/Segment Anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def predict_from_boxes(self, image, box):
        # example box: np.array([425, 600, 700, 875])
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
        #in masks [0] if false 0, true 255

        mask = masks[0].astype(np.uint8)
        return mask
    

def main():
    sam = SAM()
    image = cv2.imread("/home/fyp/image.jpg")
    box = np.array([693, 357, 858, 516])
    mask = sam.predict_from_boxes(image, box)*255
    mask = np.array(mask)
    print(np.unique(mask))
    print(mask.shape) # the shape is (1,720,1280)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.imwrite("/home/fyp/mask.jpg", mask)


if __name__ == '__main__':
    main()