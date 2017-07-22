import os
import tensorflow as tf
import numpy as np
import Image
import pickle
from object_detection.utils import label_map_util


class Obj_detector(object):
    def __init__(self, path_to_save, path_of_images, model_name=None, path_to_labels=None):
        self.path_to_save = path_to_save
        if model_name is None:
            self.model_name = 'ssd_mobilenet_v1_coco_11_06_2017'
        else:
            self.model_name = model_name
        if path_to_labels == None:
            self.path_to_labels = os.path.join(os.getcwd(), '/../object_detection/data/mscoco_label_map.pbtxt')
        else:
            self.path_to_labels = path_to_labels
        self.path_to_scpt = os.path.join(os.getcwd(), '..', 'object_detection/' + self.model_name + '/frozen_inference_graph.pb')
        self.path_of_images = path_of_images

    def fit_transform(self, list_of_paths, num_classes=90, save=True):
        detection_graph = self._make_comp_graph()
        # label_map, categories, category_index = self._make_label(num_classes)
        with detection_graph.as_default():
            with tf.device('/gpu:0'):
                with tf.Session(graph=detection_graph) as sess:
                    print(list_of_paths)
                    for image_path in list_of_paths:
                        image = Image.open(os.path.join(self.path_of_images, image_path))

                        image_np = self._load_image_into_numpy_array(image)

                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')

                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        (boxes_res, scores_res, classes_res, num_detections_res) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                        image_name = image_path.split('/')[-1].split('.')[0]
                        if save:
                            self._save(image_name, boxes_res, scores_res, classes_res, num_detections_res)
        return None

    def _make_comp_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_scpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _make_label(self, num_classes):
        label_map = label_map_util.load_labelmap(self.path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)

        category_index = label_map_util.create_category_index(categories)
        return label_map, categories, category_index

    @staticmethod
    def _load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def _save(self, image_name, boxes, scores, classes, num_detections):
        a = {}
        a['boxes'] = boxes
        a['scores'] = scores
        a['classes'] = classes
        a['num_detections'] = num_detections
        with open(os.path.join(self.path_to_save, image_name + '.pickle'), 'wb') as f:
            print(os.path.join(self.path_to_save, image_name + '.pickle'))
            pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)
