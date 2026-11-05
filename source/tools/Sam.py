import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from skimage import measure
from PySide6.QtCore import Qt, Signal
from source.Mask import paintMask, jointBox, jointMask, replaceMask, checkIntersection, intersectMask
from source.genutils import qimageToNumpyArray, cropQImage
from source.ModelWeights import ensureModel
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QPen, QBrush
from source import genutils
from source.tools.Tool import Tool
from source.Blob import Blob

from PySide6.QtCore import Qt, QObject, QPointF, QRectF, QFileInfo, QDir, Slot, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPainterPath, QPen, QBrush, QCursor, QColor
from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsItem, QGraphicsScene, QFileDialog, QGraphicsPixmapItem


sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.amg import build_all_layer_point_grids
import time

SAM_INPUT_SIZE = 1024  # SAM's own input size (long side, pixels)


def upscaleMask(mask, height, width):
    """A SAM mask dict computed on a downscaled image, as the same dict at the original size (height, width)."""
    full = cv2.resize(mask["segmentation"].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(full)
    if len(ys) == 0:
        return dict(mask, segmentation=full, area=0, bbox=[0, 0, 0, 0])
    return dict(mask, segmentation=full, area=int(full.sum()),
                bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)])


def workAreaHint(blobs, area_width):
    """
    A hint for the user when the work area may be too small for the colonies. SAM outlines a colony as one piece when it
    spans roughly 10-45 % of the work-area width (measured on a 13 cm colony: a 19 cm area gave polyp-sized pieces,
    37 cm or more the whole colony). TagLab does not know the colony, so the largest outline stands in for it: when it
    spans more than 45 % of the width the area is tight, and when it spans less than 15 % the colony was probably cut
    into pieces (a 512 px area on a colony spanning 53 % of it: largest piece 11 %; 1024 px: the colony, 33 %; 2048 px: 17 %).
    Returns the text, or None when the largest outline is in range. Blobs are in map pixels.
    """
    if not blobs or area_width <= 0:
        return None
    largest = max(max(blob.bbox[2], blob.bbox[3]) for blob in blobs) / area_width
    if 0.15 <= largest <= 0.45:
        return None
    return "The largest outline spans %d %% of the work area width. Automatic SAM outlines a colony as one piece when it " \
           "spans 10-45 %%: if you expected larger outlines, choose a larger work area (SHIFT + WHEEL) and try again." \
           % round(100 * largest)


class Sam(Tool):

    samEnded = Signal()
    

    def __init__(self, viewerplus, pick_points):
        super(Sam, self).__init__(viewerplus)

        self.viewerplus.mouseMoved.connect(self.handlemouseMove)

        #1024x1024 rect_item size
        self.width = 1024
        self.height = 1024

        self.offset = [0, 0]

        self.rect_item = None

        self.seed_nr = 0
        self.not_overlapping = []
        
        message = "<p><i>Automatic segmentation of all instances in a work-area</i></p>"
        message += "<p><b>STEP 1</b>: Choose the work-area:<br/>\
                    - SHIFT + WHEEL to change area size<br/>\
                    - SHIFT + LMB to set the area</p>"
        message += "<p><b>STEP 2</b>: Set seeds:<br/>\
                    - SHIFT + WHEEL to change the spacing</p>"
        message += "SPACEBAR to apply segmentation</p>"
        self.tool_message = f'<div style="text-align: left;">{message}</div>'

        # User defined points
        self.pick_points = pick_points

        self.work_area_item = None
        self.work_area_rect = None
        self.work_area_set = False
        self.shadow_item = None

        self.image_cropped = None
        self.image_cropped_np = None
        
        self.work_points = []
        self.num_points = 32

        self.CROSS_LINE_WIDTH = 2
        self.work_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.cyan, 'size': 8}
        self.pos_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.green, 'size': 4}
        self.increase = 5

        self.sam_net = None
        self.device = None
        self.created_blobs = []
        self.hint = None

    def activate(self):
        self.viewerplus.showMessage(self.tool_message)

    def deactivate(self):
        self.viewerplus.clearMessage()

    def handlemouseMove(self, x, y):
        """
         Sam parameters: 
            pred_iou_thresh (float): A filtering threshold in [0,1], using the model's predicted mask quality.
            stability_score_thresh (float): A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize  the model's mask predictions.
            stability_score_offset (float): The amount to shift the cutoff when calculated the stability score.
            box_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
            
        IDEA: SE ZOOM LEVEL 0 ALLORA FA TUTTA IMMAGINE 
              SE ZOOM LEVEL E' X >>  1024 AVVISA CHE è GROSSA
              SE ZOMM LEVEL +- 1024 allora prende 1024
              se zoom level << 1024 sovracampiona a 1024 (lo fa lui già mi sA) E CONTA 
            
        
        """
        #add working area

    
    def setWorkArea(self):
        """
		Set the work area based on the location of points
		"""
        # Display to GUI
        brush = QBrush(Qt.NoBrush)
        pen = QPen(Qt.DashLine)
        pen.setWidth(2)
        pen.setColor(Qt.white)
        pen.setCosmetic(True)

        # From the current view, crop the image
        # Get the bounding rect of the work area and its position
        rect = self.rect_item.boundingRect()
        self.work_area_rect = self.viewerplus.scene.addRect(rect, pen, brush)

        pos = self.rect_item.pos()      
        rect.moveTopLeft(pos)
        rect = rect.normalized()
        rect = rect.intersected(self.viewerplus.sceneRect())
        self.work_area_rect.setPos(pos)

        self.work_area_item = rect

        # Use the clipped rect's coordinates as offset, not the original position
        # This ensures correct blob placement when work area extends beyond map borders
        self.offset = [rect.left(), rect.top()]

        image_cropped = self.viewerplus.img_map.copy(rect.toRect())
        
        # Crop the image based on the work area
        self.image_cropped = image_cropped

        # Save the cropped image
        # image_cropped.save("cropped_image.png")
        
        self.image_cropped_np = qimageToNumpyArray(image_cropped)
        self.viewerplus.scene.removeItem(self.rect_item)

        # Create a semi-transparent overlay
        shadow_brush = QBrush(QColor(0, 0, 0, 75))  # Semi-transparent black
        shadow_path = QPainterPath()
        shadow_path.addRect(self.viewerplus.sceneRect())  # Cover the entire scene
        shadow_path.addRect(rect)  # Add the work area rect

        # Subtract the work area from the overlay
        shadow_path = shadow_path.simplified()

        # Add the overlay to the scene
        self.shadow_item = self.viewerplus.scene.addPath(shadow_path, QPen(Qt.NoPen), shadow_brush)

        self.work_area_set = True

    def increasePoint(self, delta):
        #increase value got from delta angle of mouse wheel
        increase = int(delta.y())/100
        # print(f'increase is {increase}')
                
        #set increase or decrease value on  wheel rotation direction
        if 0.0 < increase < 1.0:
            increase = 10
        elif -1.0 < increase < 0.0:
            increase = -10

        self.increase = increase
        print(f'self.increase is {self.increase}')
        self.setWorkPoints(self.increase)

    def setWorkPoints(self, increase = 0):

        # Reset the current points shown
        self.pick_points.reset()

        # Change the number of points to display
        self.num_points += increase
        if self.num_points < 2:
            self.num_points = 2
        elif self.num_points > 64:
            self.num_points = 64

        # Get the updated number of points
        points_tensor = build_all_layer_point_grids(self.num_points, 0, 1)[0]
        
        # Ensure tensors are on CPU for coordinate calculations
        if hasattr(points_tensor, 'cpu'):
            points_tensor = points_tensor.cpu()
        
        x_pts, y_pts = points_tensor.T

        left = self.work_area_item.left()
        top = self.work_area_item.top()

        # Change coordinates to match viewer
        x_pts = (x_pts * self.image_cropped_np.shape[1]) + left
        y_pts = (y_pts * self.image_cropped_np.shape[0]) + top

        # Add all of them to the list
        for x, y in list(zip(x_pts, y_pts)):
            self.pick_points.addPoint(x, y, self.pos_pick_style)

    def setSize(self, delta):
        #increase value got from delta angle of mouse wheel
        increase = float(delta.y()) / 10.0
        
        #set increase or decrease value on  wheel rotation direction
        if 0.0 < increase < 1.0:
            increase = 100
        elif -1.0 < increase < 0.0:
            increase = -100

        #rescale rect_item on zoom factor from wheel event
        # added *2 to mantain rectangle inside the map
        new_width = self.width + (increase)
        new_height = self.height + (increase)
        
        #limit the rectangle to 512x512 for SAM segmentation
        if new_width < 512 or new_height < 512:
            new_width = 512
            new_height = 512

        # limit the rectangle to 2048x2048 for SAM segmentation
        if new_width > 2048 or new_height > 2048:
            new_width = 2048
            new_height = 2048
  
        # print(f"rect_item width and height are {new_width, new_height}")
        if self.rect_item is not None:
            self.rect_item.setRect(0, 0, new_width, new_height)

        self.width = new_width
        self.height = new_height


    def loadNetwork(self):

        if self.sam_net is None:

            self.infoMessage.emit("Loading SAM network..")
            # add choices related to GPU MEMORY

            sam_checkpoint = "sam_vit_h_4b8939.pth"
            model_type = "vit_h"

            network_name = ensureModel(sam_checkpoint, self.viewerplus)
            if network_name is None:
                self.infoMessage.emit("SAM network not available.")
                return False

            device = genutils.torchDevice()
            print(f"SAM device: {device}")

            self.device = device
            self.sam_net = sam_model_registry[model_type](checkpoint=network_name)
            self.sam_net.to(device=self.device)
            
            print(f"SAM model loaded on device: {self.device}")

        return True       

    #remove blobs on the edge of the rectangle cursor
    def removeEdgeBlobs(self):

        if self.work_area_rect is None:
            return

        # Get the bounding rect of the rectangle and its position
        rect = self.work_area_rect.boundingRect()
        rect.moveTopLeft(self.work_area_rect.pos())
        rect = rect.normalized()

        print(f"rect coordinates are {rect.left(), rect.right(), rect.top(), rect.bottom()}")


        # margin to consider the edge TO FINETUNE!!!!
        margin = 10  
        
        # filtered_blobs = []
        blobs = self.created_blobs.copy()
        for blob in blobs:

            bbox = blob.bbox
            top = bbox[0]
            left = bbox[1]
            right = bbox[1] + bbox[2]
            bottom = bbox[0] + bbox[3]
            
            #remove blobs if they are on the edges
            if  (
                left - rect.left() < margin or
                rect.right() - right < margin or
                top - rect.top() < margin or
                rect.bottom() - bottom < margin
            ):
                self.created_blobs.remove(blob)    

    def reset(self):

        # Clean up GPU memory properly
        genutils.emptyDeviceCache()
        
        if self.sam_net is not None:
            del self.sam_net
            self.sam_net = None
            self.device = None
        
        self.image_cropped = None

        self.work_area_set = False
        self.work_area_item = None
        if self.work_area_rect is not None:
            self.viewerplus.scene.removeItem(self.work_area_rect)
            self.work_area_rect = None
        
        if self.shadow_item is not None:
            self.viewerplus.scene.removeItem(self.shadow_item)
            self.shadow_item = None
        
        self.pick_points.reset()

        if self.rect_item is not None:
            self.viewerplus.scene.addItem(self.rect_item)
        
    def handlemouseMove(self, x, y, mods=None):
        # print(f"Mouse moved to ({x}, {y})")
        if self.rect_item is not None:
            self.rect_item.setPos(x- self.width//2, y - self.height//2)
            
    #SAM segmentation on space key pressed instead of left mouse button pressed
    # def leftPressed(self, x, y, mods):

    def leftPressed(self, x, y, mods=None):

        if mods == Qt.ShiftModifier:        
            if not self.work_area_set:
                self.setWorkArea()
                self.pick_points.reset()
                self.setWorkPoints(increase = self.increase)
            else:
                self.reset()
        
    def apply(self):
        
        self.viewerplus.resetSelection()

        print(f"number of sam_seed is {self.num_points}")

        try:
            # Perform segmentation on the cropped image
            self.segment(self.image_cropped, self.num_points)
        except Exception as e:
            if not self.work_area_set:
                # box = QMessageBox()
                # box.setIcon(QMessageBox.Critical)
                # box.setText("Working area is not set.\nPlease define the working area before segmentation.")
                # box.setWindowTitle("No Working Area")
                # box.setStandardButtons(QMessageBox.Ok)	
                # box.exec()
                QMessageBox.critical(None, "Error", f"Work area is not set.\nPlease, define the area where apply the segmentation.")
                self.viewerplus.resetTools()
            else:
                box = QMessageBox()
                text = "Segmentation failed: {e}".format(e=e)
                box.setText(text)
                box.exec()
                self.viewerplus.resetTools()

        self.viewerplus.resetTools()

        # after resetTools, which clears the message panel
        if self.hint is not None:
            self.viewerplus.showMessage("<p>" + self.hint + "</p>", "warning", 20000)
            self.hint = None

    def segment(self, image, seed = 32, save_status=True):

        self.infoMessage.emit("Segmentation is ongoing..")
        self.log.emit("[TOOL][SAM] Segmentation begins..")

        if not self.loadNetwork():
            return

        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_net,

            # points_per_side = seed,
            points_per_side= seed,
            
            points_per_batch=64,
            crop_n_layers = 0,
            pred_iou_thresh = 0.88,
            stability_score_thresh=  0.95,
            stability_score_offset = 1.0,
            box_nms_thresh = 0.7,
            crop_nms_thresh = 0.7,
            min_mask_region_area = 1000,
            crop_overlap_ratio = 0.34333,
            crop_n_points_downscale_factor = 1,
            output_mode = "binary_mask"
        )

        if self.device is not None and self.device.type == "mps":
            # segment_anything hands float64 prompt points to the model; the Apple GPU has no float64
            transform = mask_generator.predictor.transform
            apply_coords = transform.apply_coords
            transform.apply_coords = lambda *a, **k: apply_coords(*a, **k).astype(np.float32)

        image = genutils.qimageToNumpyArray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start = time.time()

        # SAM resizes its input to 1024 px (long side) itself: hand it that size directly. The result is the
        # same, but a 2048 px work area takes 10 s instead of 140 s (Apple GPU), and the masks are mapped back.
        full_h, full_w = image.shape[:2]
        scale = SAM_INPUT_SIZE / max(full_h, full_w)
        if scale < 1.0:
            image = cv2.resize(image, (round(full_w * scale), round(full_h * scale)), interpolation=cv2.INTER_AREA)

        masks = mask_generator.generate(image)

        if scale < 1.0:
            masks = [upscaleMask(mask, full_h, full_w) for mask in masks]

        end = time.time()

        print(end-start)

        offx = self.offset[0]
        offy = self.offset[1]
        for mask in masks:
            bbox = mask["bbox"]
            bbox = [int(value) for value in bbox]
            segm_mask = mask["segmentation"].astype('uint8')*255
            segm_mask_crop = segm_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            blob = self.viewerplus.image.annotations.createBlobFromSingleMask(segm_mask_crop, bbox[0] + offx, bbox[1] + offy)
            self.created_blobs.append(blob)

        print(f"self.created_blob len pre is {len(self.created_blobs)}")
        genutils.removeOverlapping(self.created_blobs, self.created_blobs)
        print(f"self.created_blob len post is {len(self.created_blobs)}")
        self.removeEdgeBlobs()
        print(f"self.created_blob len post removeEdge is {len(self.created_blobs)}")
        print(f"Number of yet annotated blobs is {len(self.viewerplus.image.annotations.seg_blobs)}")

        genutils.removeOverlapping(self.created_blobs, self.viewerplus.image.annotations.seg_blobs, annotated = True)

        print(f"self.created_blob len post annotated is {len(self.created_blobs)}")
        
        kernel = np.ones((5, 5), np.uint8)
        created = self.created_blobs.copy()
        for blob in created:
            mask = blob.getMask()
            closed_mask = cv2.morphologyEx(mask.astype('uint8')*255, cv2.MORPH_CLOSE, kernel)
            closed_mask = (closed_mask > 0).astype('uint8')
            blob.updateUsingMask(blob.bbox, closed_mask)
            self.viewerplus.addBlob(blob, selected=True)

        hint = workAreaHint(created, self.work_area_item.width() if self.work_area_item is not None else 0)
        self.hint = hint
        if hint is not None:
            self.log.emit("[TOOL][SAM] " + hint)

        self.viewerplus.saveUndo()
        self.created_blobs = []
        self.samEnded.emit()

    
    #method to display the rectangle on the map
    def enable(self, enable = False):
        if enable == True:
            self.rect_item = self.viewerplus.scene.addRect(0, 0, self.width, self.height, QPen(Qt.black, 5, Qt.DotLine)) 
        else:
            if self.rect_item is not None:
                self.viewerplus.scene.removeItem(self.rect_item)
            self.rect_item = None
            # self.center_item.setVisible(False)