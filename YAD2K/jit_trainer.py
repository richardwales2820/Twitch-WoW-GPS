"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

class TrainingGenerator(object):
    # Batch size must be a multiple of number of icons
    def __init__(self, anchors, bg_path, icons_path, batch_size):
        self.anchors = anchors
        self.backgrounds = os.listdir(bg_path)
        self.icons = os.listdir(icons_path)
        self.batch_size = batch_size
        self.icons_path = icons_path
        self.bg_path = bg_path

    def training_generator(self):
        while True:
            bgs = random.sample(self.backgrounds, int(self.batch_size/len(self.icons)))
            image_batch = []
            box_batch = []

            for bg in bgs:
                bg_np = self.load_image(os.path.join(self.bg_path, bg))
                bg_height, bg_width, _ = bg_np.shape

                # Setup grid (the cell dimensions should be computable, easier to be given by user)
                cell_rows = 2
                cell_cols = 2
                cell_height = int(bg_height / cell_rows)
                cell_width = int(bg_width / cell_cols)

                cells = []
                for i in range(cell_cols):
                    for j in range(cell_rows):
                        cells.append((cell_width*i, cell_height*j))
                        
                random.shuffle(cells)
                icon_boxes = []
                for icon in self.icons:
                    icon_np = self.load_image(os.path.join(self.icons_path, icon))
                    icon_height, icon_width, _ = icon_np.shape

                    cell = cells.pop()
                    
                    start_x = random.randint(cell[0], cell[0]+cell_width-icon_width)
                    start_y = random.randint(cell[1], cell[1]+cell_height-icon_height)
                    print(icon_np.shape)
                    print(bg_np.shape)
                    # Overlay the icon onto the background
                    for x in range(start_x, start_x + icon_width):
                        for y in range(start_y, start_y + icon_height):
                            bg_np[y][x] = icon_np[y-start_y][x-start_x]
                    icon_boxes.append(np.array([icon, start_x, start_y, start_x+icon_width, start_y+icon_height], dtype=object))
                # Add BB for this image
                box_batch.append(np.array(icon_boxes))
                
                processed_image = Image.fromarray(bg_np, 'RGB').resize((416, 416), PIL.Image.BICUBIC)
                processed_image = np.array(processed_image, dtype=np.float)
                processed_image = processed_image/255.
                
                image_batch.append(processed_image)
            box_batch = self.process_box(np.array(box_batch), icon_width, icon_height)
            detectors_mask, matching_true_boxes = get_detector_mask(box_batch, self.anchors)

            yield [np.array(image_batch), box_batch, detectors_mask, matching_true_boxes], np.zeros(len(images))

    def process_box(self, boxes, width, height):
        orig_size = np.array([width, height])
        orig_size = np.expand_dims(orig_size, axis=0)
        
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        
        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        return np.array(boxes)

    def load_image(self, infilename):
        img = Image.open(infilename)
        img.load()
        data = np.asarray(img, dtype="int8")
        return data

# Args
argparser = argparse.ArgumentParser(
    description="Retrain YOLOv2 model with data generated JIT")

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-b',
    '--backgrounds_path',
    help='path to the background images that will have icons overlaid on top of',
)

argparser.add_argument(
    '-i',
    '--icons_path',
    help='path to the icons to be overlaid on top of the backgrounds'
)

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('..', 'DATA', 'underwater_classes.txt'))

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main(args):
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    backgrounds_path = os.path.expanduser(args.backgrounds_path)
    icons_path = os.path.expanduser(args.icons_path)
    batch_size = 64
    
    class_names = os.listdir(icons_path)
    
    anchors = get_anchors(anchors_path)

    anchors = YOLO_ANCHORS

    model_body, model = create_model(anchors, class_names)

    training_gen = TrainingGenerator(anchors, backgrounds_path, icons_path, batch_size)

    train(
        model,
        class_names,
        anchors,
        training_gen
    )
    image_data = training_gen.training_generator[0][0]
    draw(model_body,
        class_names,
        anchors,
        image_data,
        weights_name='trained_stage_3_best.h5',
        save_all=False)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

def train(model, class_names, anchors, training_gen):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    steps = 100
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    model.fit_generator(training_gen.training_generator(), epochs=5, steps_per_epoch=steps, callbacks=[logging])
    
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit_generator(training_gen.training_generator, epochs=30, steps_per_epoch=steps, callbacks=[logging])

    model.save_weights('trained_stage_2.h5')

    model.fit_generator(training_gen.training_generator, epochs=30, steps_per_epoch=steps, callbacks=[logging, checkpoint, early_stopping])

    model.save_weights('trained_stage_3.h5')

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()



if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
