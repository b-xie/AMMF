import numpy as np

import tensorflow as tf

from ammf.builders import proposal_generation_phase_builder
from ammf.builders import proposal_generation_phase_builder2
from ammf.builders import ammf_loss_builder
from ammf.core import anchor_projector
from ammf.core import anchor_encoder
from ammf.core import box_3d_encoder
from ammf.core import box_8c_encoder
from ammf.core import box_4c_encoder

from ammf.core import box_list
from ammf.core import box_list_ops

from ammf.core import model
from ammf.core import orientation_encoder
from ammf.core.models.p1_model import p1Model
#0601
# So the usage of ground plane is very limited, it should be pretty easy to change it to my input

class ammfModel(model.DetectionModel):
    ##############################
    #1. Keys for Predictions
    ##############################
    #1.1 Mini batch (mb) ground truth
    PRED_MB_CLASSIFICATIONS_GT = 'ammf_mb_classifications_gt'
    PRED_MB_OFFSETS_GT = 'ammf_mb_offsets_gt'
    PRED_MB_ORIENTATIONS_GT = 'ammf_mb_orientations_gt'

    #1.2 Mini batch (mb) predictions
    PRED_MB_CLASSIFICATION_LOGITS = 'ammf_mb_classification_logits'
    PRED_MB_CLASSIFICATION_SOFTMAX = 'ammf_mb_classification_softmax'
    PRED_MB_OFFSETS = 'ammf_mb_offsets'
    PRED_MB_ANGLE_VECTORS = 'ammf_mb_angle_vectors'

    #1.3 Top predictions after BEV NMS
    PRED_TOP_CLASSIFICATION_LOGITS = 'ammf_top_classification_logits'
    PRED_TOP_CLASSIFICATION_SOFTMAX = 'ammf_top_classification_softmax'

    PRED_TOP_PREDICTION_ANCHORS = 'ammf_top_prediction_anchors'
    PRED_TOP_PREDICTION_BOXES_3D = 'ammf_top_prediction_boxes_3d'
    PRED_TOP_ORIENTATIONS = 'ammf_top_orientations'

    #1.4 Other box representations
    PRED_TOP_BOXES_8C = 'ammf_top_regressed_boxes_8c'
    PRED_TOP_BOXES_4C = 'ammf_top_prediction_boxes_4c'

    #1.5 Mini batch (mb) predictions (for debugging)
    PRED_MB_MASK = 'ammf_mb_mask'
    PRED_MB_POS_MASK = 'ammf_mb_pos_mask'
    PRED_MB_ANCHORS_GT = 'ammf_mb_anchors_gt'
    PRED_MB_CLASS_INDICES_GT = 'ammf_mb_gt_classes'

    #1.6 All predictions (for debugging)
    PRED_ALL_CLASSIFICATIONS = 'ammf_classifications'
    PRED_ALL_OFFSETS = 'ammf_offsets'
    PRED_ALL_ANGLE_VECTORS = 'ammf_angle_vectors'

    PRED_MAX_IOUS = 'ammf_max_ious'
    PRED_ALL_IOUS = 'ammf_anchor_ious'

    ##############################
    #2 Keys for Loss
    ##############################
    LOSS_FINAL_CLASSIFICATION = 'ammf_classification_loss'
    LOSS_FINAL_REGRESSION = 'ammf_regression_loss'

    # (for debugging)
    LOSS_FINAL_ORIENTATION = 'ammf_orientation_loss'
    LOSS_FINAL_LOCALIZATION = 'ammf_localization_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        #1. Sets model configs (_config)
        super(ammfModel, self).__init__(model_config)

        self.dataset = dataset

        # Dataset config
        self._num_final_classes = self.dataset.num_classes + 1

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = [input_config.img_depth]
       
        #2. p2 config
        p2_config = self._config.p2_config
        self.p2_proposal_roi_crop_size = \
            [p2_config.p2_proposal_roi_crop_size] * 2
        self.p2_positive_selection = p2_config.p2_positive_selection
        self.p2_nms_size = p2_config.p2_nms_size
        self.p2_nms_iou_threshold = p2_config.p2_nms_iou_thresh
        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._box_rep = p2_config.p2_box_representation
        
        #2. p3 config
        p3_config = self._config.p3_config
        self.p3_proposal_roi_crop_size = \
            [p3_config.p3_proposal_roi_crop_size] * 2
        self.p3_positive_selection = p3_config.p3_positive_selection
        self.p3_nms_size = p3_config.p3_nms_size
        self.p3_nms_iou_threshold = p3_config.p3_nms_iou_thresh
        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._box_rep = p3_config.p3_box_representation


        if self._box_rep not in ['box_3d', 'box_8c', 'box_8co',
                                 'box_4c', 'box_4ca']:
            raise ValueError('Invalid box representation', self._box_rep)

        #3.Create the p1Model
        self._p1_model = p1Model(model_config, train_val_test, dataset)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test
        self._is_training = (self._train_val_test == 'train')

        self.sample_info = {}

    def build(self):
        p1_model = self._p1_model

        #1. Share the same prediction dict as p1
        prediction_dict = p1_model.build()

        #already after a nms
        top_anchors = prediction_dict[p1Model.PRED_TOP_ANCHORS]
        ground_plane = p1_model.placeholders[p1Model.PL_GROUND_PLANE]

        class_labels = p1_model.placeholders[p1Model.PL_LABEL_CLASSES]
        
        #2.p2 projection
        with tf.variable_scope('p2_projection'):

            if self._config.expand_proposals_xz > 0.0:#?why
                #  not active for peds
                expand_length = self._config.expand_proposals_xz

                # Expand anchors along x and z
                with tf.variable_scope('expand_xz'):
                    expanded_dim_x = top_anchors[:, 3] + expand_length
                    expanded_dim_z = top_anchors[:, 5] + expand_length

                    expanded_anchors = tf.stack([
                        top_anchors[:, 0],
                        top_anchors[:, 1],
                        top_anchors[:, 2],
                        expanded_dim_x,
                        top_anchors[:, 4],
                        expanded_dim_z
                    ], axis=1)

                ammf_projection_in = expanded_anchors

            else:
                ammf_projection_in = top_anchors

            with tf.variable_scope('bev'):
                # Project top anchors into bev and image spaces
                bev_proposal_boxes, bev_proposal_boxes_norm = \
                    anchor_projector.project_to_bev(
                        ammf_projection_in,
                        self.dataset.kitti_utils.bev_extents)

                # Reorder projected boxes into [y1, x1, y2, x2]
                bev_proposal_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        bev_proposal_boxes)
                bev_proposal_boxes_norm_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        bev_proposal_boxes_norm)

            with tf.variable_scope('img'):
                image_shape = tf.cast(tf.shape(
                    p1_model.placeholders[p1Model.PL_IMG_INPUT])[0:2],
                    tf.float32)
                img_proposal_boxes, img_proposal_boxes_norm = \
                    anchor_projector.tf_project_to_image_space(
                        ammf_projection_in,
                        p1_model.placeholders[p1Model.PL_CALIB_P2],
                        image_shape)
                # Only reorder the normalized img
                img_proposal_boxes_norm_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        img_proposal_boxes_norm)

        bev_feature_maps = p1_model.bev_feature_maps
        img_feature_maps = p1_model.img_feature_maps

        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):
        #  act like dropout layer?

            with tf.variable_scope('ammf_path_drop'):

                img_mask = p1_model.img_path_drop_mask
                bev_mask = p1_model.bev_path_drop_mask

                img_feature_maps = tf.multiply(img_feature_maps,
                                               img_mask)

                bev_feature_maps = tf.multiply(bev_feature_maps,
                                               bev_mask)
        else:
            bev_mask = tf.constant(1.0)
            img_mask = tf.constant(1.0)

        #3. ROI Pooling
        with tf.variable_scope('ammf_roi_pooling'):
            def get_box_indices(boxes):
                proposals_shape = boxes.get_shape().as_list()
                if any(dim is None for dim in proposals_shape):
                    proposals_shape = tf.shape(boxes)
                ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                multiplier = tf.expand_dims(
                    tf.range(start=0, limit=proposals_shape[0]), 1)
                return tf.reshape(ones_mat * multiplier, [-1])

            bev_boxes_norm_batches = tf.expand_dims(
                bev_proposal_boxes_norm, axis=0)

            # These should be all 0's since there is only 1 image
            tf_box_indices = get_box_indices(bev_boxes_norm_batches)

            # Do ROI Pooling on BEV
            bev_rois = tf.image.crop_and_resize(
                bev_feature_maps,
                bev_proposal_boxes_norm_tf_order,
                tf_box_indices,
                self.p2_proposal_roi_crop_size, #  7 for peds
                name='bev_rois')
            # Do ROI Pooling on image
            img_rois = tf.image.crop_and_resize(
                img_feature_maps,
                img_proposal_boxes_norm_tf_order,
                tf_box_indices,
                self.p2_proposal_roi_crop_size,
                name='img_rois')
            #bqx:bev_rois and img_rois used next inputs!!
            self.bev_rois=bev_rois
            self.img_rois=img_rois
        
        #4. Fully connected layers (Box Predictor)
        ammf_layers_config = self.model_config.layers_config.ammf_config

        fc_output_layers = \
            proposal_generation_phase_builder.build(
                layers_config=ammf_layers_config,
                input_rois=[bev_rois, img_rois],
                input_weights=[bev_mask, img_mask],
                num_final_classes=self._num_final_classes,
                box_rep=self._box_rep,
                top_anchors=top_anchors,
                ground_plane=ground_plane,
                is_training=self._is_training)

        all_cls_logits = \
            fc_output_layers[proposal_generation_phase_builder.KEY_CLS_LOGITS]
        all_offsets = fc_output_layers[proposal_generation_phase_builder.KEY_OFFSETS]

        # This may be None
        all_angle_vectors = \
            fc_output_layers.get(proposal_generation_phase_builder.KEY_ANGLE_VECTORS)

        with tf.variable_scope('softmax'):
            all_cls_softmax = tf.nn.softmax(
                all_cls_logits)

        ######################################################
        #5. Subsample mini_batch for the loss function
        ######################################################
        # Get the ground truth tensors
        anchors_gt = p1_model.placeholders[p1Model.PL_LABEL_ANCHORS]
        if self._box_rep in ['box_3d', 'box_4ca']:
            boxes_3d_gt = p1_model.placeholders[p1Model.PL_LABEL_BOXES_3D]
            orientations_gt = boxes_3d_gt[:, 6]
        elif self._box_rep in ['box_8c', 'box_8co', 'box_4c']:
            boxes_3d_gt = p1_model.placeholders[p1Model.PL_LABEL_BOXES_3D]
        else:
            raise NotImplementedError('Ground truth tensors not implemented')

        # Project anchor_gts to 2D bev
        with tf.variable_scope('ammf_gt_projection'):
            bev_anchor_boxes_gt, _ = anchor_projector.project_to_bev(
                anchors_gt, self.dataset.kitti_utils.bev_extents)

            bev_anchor_boxes_gt_tf_order = \
                anchor_projector.reorder_projected_boxes(bev_anchor_boxes_gt)

        with tf.variable_scope('ammf_box_list'):
            # Convert to box_list format
            anchor_box_list_gt = box_list.BoxList(bev_anchor_boxes_gt_tf_order)
            anchor_box_list = box_list.BoxList(bev_proposal_boxes_tf_order)

        mb_mask, mb_class_label_indices, mb_gt_indices = \
            self.sample_mini_batch(
                anchor_box_list_gt=anchor_box_list_gt,
                anchor_box_list=anchor_box_list,
                class_labels=class_labels)

        # Create classification one_hot vector
        with tf.variable_scope('ammf_one_hot_classes'):
            mb_classification_gt = tf.one_hot(
                mb_class_label_indices,
                depth=self._num_final_classes,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=(self._config.label_smoothing_epsilon /
                           self.dataset.num_classes))

        # TODO: Don't create a mini batch in test mode
        # Mask predictions
        with tf.variable_scope('ammf_apply_mb_mask'):
            # Classification
            mb_classifications_logits = tf.boolean_mask(
                all_cls_logits, mb_mask)
            mb_classifications_softmax = tf.boolean_mask(
                all_cls_softmax, mb_mask)

            # Offsets
            mb_offsets = tf.boolean_mask(all_offsets, mb_mask)

            # Angle Vectors
            if all_angle_vectors is not None:
                mb_angle_vectors = tf.boolean_mask(all_angle_vectors, mb_mask)
            else:
                mb_angle_vectors = None

        # Encode anchor offsets
        with tf.variable_scope('ammf_encode_mb_anchors'):
            mb_anchors = tf.boolean_mask(top_anchors, mb_mask)

            if self._box_rep == 'box_3d':
                # Gather corresponding ground truth anchors for each mb sample
                mb_anchors_gt = tf.gather(anchors_gt, mb_gt_indices)
                mb_offsets_gt = anchor_encoder.tf_anchor_to_offset(
                    mb_anchors, mb_anchors_gt)

                # Gather corresponding ground truth orientation for each
                # mb sample
                mb_orientations_gt = tf.gather(orientations_gt,
                                               mb_gt_indices)
            elif self._box_rep in ['box_8c', 'box_8co']:

                # Get boxes_3d ground truth mini-batch and convert to box_8c
                mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                if self._box_rep == 'box_8c':
                    mb_boxes_8c_gt = \
                        box_8c_encoder.tf_box_3d_to_box_8c(mb_boxes_3d_gt)
                elif self._box_rep == 'box_8co':
                    mb_boxes_8c_gt = \
                        box_8c_encoder.tf_box_3d_to_box_8co(mb_boxes_3d_gt)

                # Convert proposals: anchors -> box_3d -> box8c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(top_anchors, fix_lw=True)
                proposal_boxes_8c = \
                    box_8c_encoder.tf_box_3d_to_box_8c(proposal_boxes_3d)

                # Get mini batch offsets
                mb_boxes_8c = tf.boolean_mask(proposal_boxes_8c, mb_mask)
                mb_offsets_gt = box_8c_encoder.tf_box_8c_to_offsets(
                    mb_boxes_8c, mb_boxes_8c_gt)

                # Flatten the offsets to a (N x 24) vector
                mb_offsets_gt = tf.reshape(mb_offsets_gt, [-1, 24])

            elif self._box_rep in ['box_4c', 'box_4ca']:

                # Get ground plane for box_4c conversion
                ground_plane = self._p1_model.placeholders[
                    self._p1_model.PL_GROUND_PLANE]

                # Convert gt boxes_3d -> box_4c
                mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                mb_boxes_4c_gt = box_4c_encoder.tf_box_3d_to_box_4c(
                    mb_boxes_3d_gt, ground_plane)

                # Convert proposals: anchors -> box_3d -> box_4c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(top_anchors, fix_lw=True)
                proposal_boxes_4c = \
                    box_4c_encoder.tf_box_3d_to_box_4c(proposal_boxes_3d,
                                                       ground_plane)

                # Get mini batch
                mb_boxes_4c = tf.boolean_mask(proposal_boxes_4c, mb_mask)
                mb_offsets_gt = box_4c_encoder.tf_box_4c_to_offsets(
                    mb_boxes_4c, mb_boxes_4c_gt)

                if self._box_rep == 'box_4ca':
                    # Gather corresponding ground truth orientation for each
                    # mb sample
                    mb_orientations_gt = tf.gather(orientations_gt,
                                                   mb_gt_indices)

            else:
                raise NotImplementedError(
                    'Anchor encoding not implemented for', self._box_rep)

        ######################################################
        #6. ROI summary images
        # WZN: very useful, show the pooled RoI in raw data
        ######################################################
        ammf_mini_batch_size = \
            self.dataset.kitti_utils.mini_batch_utils.ammf_mini_batch_size
        with tf.variable_scope('bev_ammf_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(
                bev_proposal_boxes_norm_tf_order, mb_mask)
            mb_bev_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = tf.image.crop_and_resize(
                self._p1_model._bev_preprocessed,
                mb_bev_anchors_norm,
                mb_bev_box_indices,
                (32, 32))
            #bqx:
            self.bev_input_rois=bev_input_rois
            bev_input_roi_summary_images = tf.split(
                bev_input_rois, self._bev_depth, axis=3)
            tf.summary.image('bev_ammf_rois',
                             bev_input_roi_summary_images[-1],
                             max_outputs=ammf_mini_batch_size)


        with tf.variable_scope('img_ammf_rois'):
            # ROIs on image input
            mb_img_anchors_norm = tf.boolean_mask(
                img_proposal_boxes_norm_tf_order, mb_mask)
            mb_img_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

            # Do test ROI pooling on mini batch
            img_input_rois = tf.image.crop_and_resize(
                self._p1_model._img_preprocessed,
                mb_img_anchors_norm,
                mb_img_box_indices,
                (32, 32))

            #bqx:
            self.img_input_rois=img_input_rois

            tf.summary.image('img_ammf_rois',
                             img_input_rois,
                             max_outputs=ammf_mini_batch_size)

        ######################################################
        #7. Final Predictions
        ######################################################
        # Get orientations from angle vectors
        if all_angle_vectors is not None:
            with tf.variable_scope('ammf_orientation'):
                all_orientations = \
                    orientation_encoder.tf_angle_vector_to_orientation(
                        all_angle_vectors)

        # Apply offsets to regress proposals
        with tf.variable_scope('ammf_regression'):
            if self._box_rep == 'box_3d':
                prediction_anchors = \
                    anchor_encoder.offset_to_anchor(top_anchors,
                                                    all_offsets)

            elif self._box_rep in ['box_8c', 'box_8co']:
                # Reshape the 24-dim regressed offsets to (N x 3 x 8)
                reshaped_offsets = tf.reshape(all_offsets,
                                              [-1, 3, 8])
                # Given the offsets, get the boxes_8c
                prediction_boxes_8c = \
                    box_8c_encoder.tf_offsets_to_box_8c(proposal_boxes_8c,
                                                        reshaped_offsets)
                # Convert corners back to box3D
                prediction_boxes_3d = \
                    box_8c_encoder.box_8c_to_box_3d(prediction_boxes_8c)

                # Convert the box_3d to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            elif self._box_rep in ['box_4c', 'box_4ca']:
                # Convert predictions box_4c -> box_3d
                prediction_boxes_4c = \
                    box_4c_encoder.tf_offsets_to_box_4c(proposal_boxes_4c,
                                                        all_offsets)

                prediction_boxes_3d = \
                    box_4c_encoder.tf_box_4c_to_box_3d(prediction_boxes_4c,
                                                       ground_plane)

                # Convert to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            else:
                raise NotImplementedError('Regression not implemented for',
                                          self._box_rep)

        # Apply Non-oriented NMS in BEV
        with tf.variable_scope('ammf_nms'):
            bev_extents = self.dataset.kitti_utils.bev_extents

            with tf.variable_scope('bev_projection'):
                # Project predictions into BEV
                ammf_bev_boxes, _ = anchor_projector.project_to_bev(
                    prediction_anchors, bev_extents)
                ammf_bev_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        ammf_bev_boxes)

            # Get top score from second column onward
            all_top_scores = tf.reduce_max(all_cls_logits[:, 1:], axis=1)
            all_top_scores = tf.reduce_max(all_cls_logits[:, 1:], axis=1)
            print(all_top_scores.shape)
            print("all_top_scores1")

            # Apply NMS in BEV
            nms_indices = tf.image.non_max_suppression(
                ammf_bev_boxes_tf_order,
                all_top_scores,
                max_output_size=self.p2_nms_size,
                iou_threshold=self.p2_nms_iou_threshold)

            # Gather predictions from NMS indices
            top_classification_logits = tf.gather(all_cls_logits,
                                                  nms_indices)
            top_classification_softmax = tf.gather(all_cls_softmax,
                                                   nms_indices)
            top_prediction_anchors = tf.gather(prediction_anchors,
                                               nms_indices)

            if self._box_rep == 'box_3d':
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            elif self._box_rep in ['box_8c', 'box_8co']:
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_8c = tf.gather(
                    prediction_boxes_8c, nms_indices)

            elif self._box_rep == 'box_4c':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)

            elif self._box_rep == 'box_4ca':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            else:
                raise NotImplementedError('NMS gather not implemented for',
                                          self._box_rep)


        # p3_projection
        with tf.variable_scope('p3_projection'):

            if self._config.expand_proposals_xz > 0.0:
                #  not active for peds
                expand_length = self._config.expand_proposals_xz

                # Expand anchors along x and z
                with tf.variable_scope('expand_xz'):
                    expanded_dim_x = top_prediction_anchors[:, 3] + expand_length
                    expanded_dim_z = top_prediction_anchors[:, 5] + expand_length

                    expanded_anchors = tf.stack([
                        top_prediction_anchors[:, 0],
                        top_prediction_anchors[:, 1],
                        top_prediction_anchors[:, 2],
                        expanded_dim_x,
                        top_prediction_anchors[:, 4],
                        expanded_dim_z
                    ], axis=1)

                ammf_projection_in = expanded_anchors

            else:
                ammf_projection_in = top_prediction_anchors

            with tf.variable_scope('bev'):
                # Project top anchors into bev and image spaces
                bev_proposal_boxes, bev_proposal_boxes_norm = \
                    anchor_projector.project_to_bev(
                        ammf_projection_in,
                        self.dataset.kitti_utils.bev_extents)

                # Reorder projected boxes into [y1, x1, y2, x2]
                bev_proposal_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        bev_proposal_boxes)
                bev_proposal_boxes_norm_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        bev_proposal_boxes_norm)

            with tf.variable_scope('img'):
                image_shape = tf.cast(tf.shape(
                    p1_model.placeholders[p1Model.PL_IMG_INPUT])[0:2],
                    tf.float32)
                img_proposal_boxes, img_proposal_boxes_norm = \
                    anchor_projector.tf_project_to_image_space(
                        ammf_projection_in,
                        p1_model.placeholders[p1Model.PL_CALIB_P2],
                        image_shape)
                # Only reorder the normalized img
                img_proposal_boxes_norm_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        img_proposal_boxes_norm)

        bev_feature_maps = bev_rois
        img_feature_maps = img_rois

        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):
        #  act like dropout layer?

            with tf.variable_scope('ammf_path_drop'):

                img_mask = p1_model.img_path_drop_mask
                bev_mask = p1_model.bev_path_drop_mask

                img_feature_maps = tf.multiply(img_feature_maps,
                                               img_mask)

                bev_feature_maps = tf.multiply(bev_feature_maps,
                                               bev_mask)
        else:
            bev_mask = tf.constant(1.0)
            img_mask = tf.constant(1.0)

        # ROI Pooling
        with tf.variable_scope('ammf_roi_pooling'):
            def get_box_indices(boxes):
                proposals_shape = boxes.get_shape().as_list()
                if any(dim is None for dim in proposals_shape):
                    proposals_shape = tf.shape(boxes)
                ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                multiplier = tf.expand_dims(
                    tf.range(start=0, limit=proposals_shape[0]), 1)
                return tf.reshape(ones_mat * multiplier, [-1])

            bev_boxes_norm_batches = tf.expand_dims(
                bev_proposal_boxes_norm, axis=0)

            # These should be all 0's since there is only 1 image
            tf_box_indices = get_box_indices(bev_boxes_norm_batches)

            # Do ROI Pooling on BEV
            bev_rois = tf.image.crop_and_resize(
                bev_feature_maps,
                bev_proposal_boxes_norm_tf_order,
                tf_box_indices,
                self.p3_proposal_roi_crop_size, #  7 for peds
                name='bev_rois')
            # Do ROI Pooling on image
            img_rois = tf.image.crop_and_resize(
                img_feature_maps,
                img_proposal_boxes_norm_tf_order,
                tf_box_indices,
                self.p3_proposal_roi_crop_size,
                name='img_rois')
        
        # Fully connected layers (Box Predictor)
        ammf_layers_config = self.model_config.layers_config.ammf_config

        fc_output_layers2 = \
            proposal_generation_phase_builder2.build(
                layers_config=ammf_layers_config,
                input_rois=[bev_rois, img_rois],
                input_weights=[bev_mask, img_mask],
                num_final_classes=self._num_final_classes,
                box_rep=self._box_rep,
                top_anchors=top_prediction_anchors,
                ground_plane=ground_plane,
                is_training=self._is_training)
        

        all_cls_logits = \
            fc_output_layers2[proposal_generation_phase_builder2.KEY_CLS_LOGITS]
        all_offsets = fc_output_layers2[proposal_generation_phase_builder2.KEY_OFFSETS]

        # This may be None
        all_angle_vectors = \
            fc_output_layers2.get(proposal_generation_phase_builder2.KEY_ANGLE_VECTORS)

        with tf.variable_scope('softmax'):
            all_cls_softmax = tf.nn.softmax(
                all_cls_logits)

        ######################################################
        # Subsample mini_batch for the loss function
        ######################################################
        # Get the ground truth tensors
        anchors_gt = p1_model.placeholders[p1Model.PL_LABEL_ANCHORS]
        if self._box_rep in ['box_3d', 'box_4ca']:
            boxes_3d_gt = p1_model.placeholders[p1Model.PL_LABEL_BOXES_3D]
            orientations_gt = boxes_3d_gt[:, 6]
        elif self._box_rep in ['box_8c', 'box_8co', 'box_4c']:
            boxes_3d_gt = p1_model.placeholders[p1Model.PL_LABEL_BOXES_3D]
        else:
            raise NotImplementedError('Ground truth tensors not implemented')

        # Project anchor_gts to 2D bev
        with tf.variable_scope('ammf_gt_projection'):
            bev_anchor_boxes_gt, _ = anchor_projector.project_to_bev(
                anchors_gt, self.dataset.kitti_utils.bev_extents)

            bev_anchor_boxes_gt_tf_order = \
                anchor_projector.reorder_projected_boxes(bev_anchor_boxes_gt)

        with tf.variable_scope('ammf_box_list'):
            # Convert to box_list format
            anchor_box_list_gt = box_list.BoxList(bev_anchor_boxes_gt_tf_order)
            anchor_box_list = box_list.BoxList(bev_proposal_boxes_tf_order)

        mb_mask, mb_class_label_indices, mb_gt_indices = \
            self.sample_mini_batch(
                anchor_box_list_gt=anchor_box_list_gt,
                anchor_box_list=anchor_box_list,
                class_labels=class_labels)

        # Create classification one_hot vector
        with tf.variable_scope('ammf_one_hot_classes'):
            mb_classification_gt = tf.one_hot(
                mb_class_label_indices,
                depth=self._num_final_classes,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=(self._config.label_smoothing_epsilon /
                           self.dataset.num_classes))

        # TODO: Don't create a mini batch in test mode
        # Mask predictions
        with tf.variable_scope('ammf_apply_mb_mask'):
            # Classification
            mb_classifications_logits = tf.boolean_mask(
                all_cls_logits, mb_mask)
            mb_classifications_softmax = tf.boolean_mask(
                all_cls_softmax, mb_mask)

            # Offsets
            mb_offsets = tf.boolean_mask(all_offsets, mb_mask)

            # Angle Vectors
            if all_angle_vectors is not None:
                mb_angle_vectors = tf.boolean_mask(all_angle_vectors, mb_mask)
            else:
                mb_angle_vectors = None

        # Encode anchor offsets
        with tf.variable_scope('ammf_encode_mb_anchors'):
            mb_anchors = tf.boolean_mask(top_prediction_anchors, mb_mask)

            if self._box_rep == 'box_3d':
                # Gather corresponding ground truth anchors for each mb sample
                mb_anchors_gt = tf.gather(anchors_gt, mb_gt_indices)
                mb_offsets_gt = anchor_encoder.tf_anchor_to_offset(
                    mb_anchors, mb_anchors_gt)

                # Gather corresponding ground truth orientation for each
                # mb sample
                mb_orientations_gt = tf.gather(orientations_gt,
                                               mb_gt_indices)
            elif self._box_rep in ['box_8c', 'box_8co']:

                # Get boxes_3d ground truth mini-batch and convert to box_8c
                mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                if self._box_rep == 'box_8c':
                    mb_boxes_8c_gt = \
                        box_8c_encoder.tf_box_3d_to_box_8c(mb_boxes_3d_gt)
                elif self._box_rep == 'box_8co':
                    mb_boxes_8c_gt = \
                        box_8c_encoder.tf_box_3d_to_box_8co(mb_boxes_3d_gt)

                # Convert proposals: anchors -> box_3d -> box8c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(top_prediction_anchors, fix_lw=True)
                proposal_boxes_8c = \
                    box_8c_encoder.tf_box_3d_to_box_8c(proposal_boxes_3d)

                # Get mini batch offsets
                mb_boxes_8c = tf.boolean_mask(proposal_boxes_8c, mb_mask)
                mb_offsets_gt = box_8c_encoder.tf_box_8c_to_offsets(
                    mb_boxes_8c, mb_boxes_8c_gt)

                # Flatten the offsets to a (N x 24) vector
                mb_offsets_gt = tf.reshape(mb_offsets_gt, [-1, 24])

            elif self._box_rep in ['box_4c', 'box_4ca']:

                # Get ground plane for box_4c conversion
                ground_plane = self._p1_model.placeholders[
                    self._p1_model.PL_GROUND_PLANE]

                # Convert gt boxes_3d -> box_4c
                mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                mb_boxes_4c_gt = box_4c_encoder.tf_box_3d_to_box_4c(
                    mb_boxes_3d_gt, ground_plane)

                # Convert proposals: anchors -> box_3d -> box_4c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(prediction_anchors, fix_lw=True)
                proposal_boxes_4c = \
                    box_4c_encoder.tf_box_3d_to_box_4c(proposal_boxes_3d,
                                                       ground_plane)

                # Get mini batch
                mb_boxes_4c = tf.boolean_mask(proposal_boxes_4c, mb_mask)
                mb_offsets_gt = box_4c_encoder.tf_box_4c_to_offsets(
                    mb_boxes_4c, mb_boxes_4c_gt)

                if self._box_rep == 'box_4ca':
                    # Gather corresponding ground truth orientation for each
                    # mb sample
                    mb_orientations_gt = tf.gather(orientations_gt,
                                                   mb_gt_indices)

            else:
                raise NotImplementedError(
                    'Anchor encoding not implemented for', self._box_rep)

        ######################################################
        # ROI summary images
        # WZN: very useful, show the pooled RoI in raw data
        ######################################################
    
        ammf_mini_batch_size = \
            self.dataset.kitti_utils.mini_batch_utils.ammf_mini_batch_size
        with tf.variable_scope('bev_ammf_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(
                bev_proposal_boxes_norm_tf_order, mb_mask)
            mb_bev_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = tf.image.crop_and_resize(
                bev_rois,
                mb_bev_anchors_norm,
                mb_bev_box_indices,
                (30, 30))

            bev_input_roi_summary_images = tf.split(
                bev_input_rois, self._bev_depth, axis=1)
            tf.summary.image('bev_ammf_rois',
                             bev_input_roi_summary_images[-1],
                             max_outputs=ammf_mini_batch_size)
       
        with tf.variable_scope('img_ammf_rois'):
            # ROIs on image input
            mb_img_anchors_norm = tf.boolean_mask(
                img_proposal_boxes_norm_tf_order, mb_mask)
            mb_img_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

            # Do test ROI pooling on mini batch
            img_input_rois = tf.image.crop_and_resize(
                img_rois,
                mb_img_anchors_norm,
                mb_img_box_indices,
                (30, 30))

            tf.summary.image('img_ammf_rois',
                             img_input_rois,
                             max_outputs=ammf_mini_batch_size)
  
        ######################################################
        # Final Predictions
        ######################################################
        # Get orientations from angle vectors
        if all_angle_vectors is not None:
            with tf.variable_scope('ammf_orientation'):
                all_orientations = \
                    orientation_encoder.tf_angle_vector_to_orientation(
                        all_angle_vectors)

        # Apply offsets to regress proposals
        with tf.variable_scope('ammf_regression'):
            if self._box_rep == 'box_3d':
                prediction_anchors = \
                    anchor_encoder.offset_to_anchor(top_prediction_anchors,
                                                    all_offsets)

            elif self._box_rep in ['box_8c', 'box_8co']:
                # Reshape the 24-dim regressed offsets to (N x 3 x 8)
                reshaped_offsets = tf.reshape(all_offsets,
                                              [-1, 3, 8])
                # Given the offsets, get the boxes_8c
                prediction_boxes_8c = \
                    box_8c_encoder.tf_offsets_to_box_8c(proposal_boxes_8c,
                                                        reshaped_offsets)
                # Convert corners back to box3D
                prediction_boxes_3d = \
                    box_8c_encoder.box_8c_to_box_3d(prediction_boxes_8c)

                # Convert the box_3d to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            elif self._box_rep in ['box_4c', 'box_4ca']:
                # Convert predictions box_4c -> box_3d


                prediction_boxes_4c = \
                    box_4c_encoder.tf_offsets_to_box_4c(proposal_boxes_4c,
                                                        all_offsets)

                prediction_boxes_3d = \
                    box_4c_encoder.tf_box_4c_to_box_3d(prediction_boxes_4c,
                                                       ground_plane)

                # Convert to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            else:
                raise NotImplementedError('Regression not implemented for',
                                          self._box_rep)

        # Apply Non-oriented NMS in BEV
        with tf.variable_scope('ammf_nms'):
            bev_extents = self.dataset.kitti_utils.bev_extents

            with tf.variable_scope('bev_projection'):
                # Project predictions into BEV
                ammf_bev_boxes, _ = anchor_projector.project_to_bev(
                    prediction_anchors, bev_extents)
                ammf_bev_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        ammf_bev_boxes)

            # Get top score from second column onward
            all_top_scores = tf.reduce_max(all_cls_logits[:, 1:], axis=1)
            print(all_top_scores.shape)

            # Apply NMS in BEV
            nms_indices = tf.image.non_max_suppression(
                ammf_bev_boxes_tf_order,
                all_top_scores,
                max_output_size=self.p3_nms_size,
                iou_threshold=self.p3_nms_iou_threshold)

            # Gather predictions from NMS indices
            top_classification_logits = tf.gather(all_cls_logits,
                                                  nms_indices)
            top_classification_softmax = tf.gather(all_cls_softmax,
                                                   nms_indices)
            top_prediction_anchors = tf.gather(prediction_anchors,
                                               nms_indices)

            if self._box_rep == 'box_3d':
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            elif self._box_rep in ['box_8c', 'box_8co']:
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_8c = tf.gather(
                    prediction_boxes_8c, nms_indices)

            elif self._box_rep == 'box_4c':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)

            elif self._box_rep == 'box_4ca':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            else:
                raise NotImplementedError('NMS gather not implemented for',
                                          self._box_rep)

                                 

        #  The prediction already contains all from the p1_model, now add others from p2 and p3 model
        if self._train_val_test in ['train', 'val']:
            # Additional entries are added to the shared prediction_dict
            # Mini batch predictions
            prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS] = \
                mb_classifications_logits
            prediction_dict[self.PRED_MB_CLASSIFICATION_SOFTMAX] = \
                mb_classifications_softmax
            prediction_dict[self.PRED_MB_OFFSETS] = mb_offsets

            # Mini batch ground truth
            prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT] = \
                mb_classification_gt
            prediction_dict[self.PRED_MB_OFFSETS_GT] = mb_offsets_gt

            # Top NMS predictions
            prediction_dict[self.PRED_TOP_CLASSIFICATION_LOGITS] = \
                top_classification_logits
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax

            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

            # Mini batch predictions (for debugging)
            prediction_dict[self.PRED_MB_MASK] = mb_mask
            # prediction_dict[self.PRED_MB_POS_MASK] = mb_pos_mask
            prediction_dict[self.PRED_MB_CLASS_INDICES_GT] = \
                mb_class_label_indices

            # All predictions (for debugging)
            prediction_dict[self.PRED_ALL_CLASSIFICATIONS] = \
                all_cls_logits
            prediction_dict[self.PRED_ALL_OFFSETS] = all_offsets

            # Path drop masks (for debugging)
            prediction_dict['bev_mask'] = bev_mask
            prediction_dict['img_mask'] = img_mask

        else:
            # self._train_val_test == 'test'
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax
            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

        if self._box_rep == 'box_3d':
            prediction_dict[self.PRED_MB_ANCHORS_GT] = mb_anchors_gt
            prediction_dict[self.PRED_MB_ORIENTATIONS_GT] = mb_orientations_gt
            prediction_dict[self.PRED_MB_ANGLE_VECTORS] = mb_angle_vectors

            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orientations

            # For debugging
            prediction_dict[self.PRED_ALL_ANGLE_VECTORS] = all_angle_vectors

        elif self._box_rep in ['box_8c', 'box_8co']:
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d

            # Store the corners before converting for visualization purposes
            prediction_dict[self.PRED_TOP_BOXES_8C] = top_prediction_boxes_8c

        elif self._box_rep == 'box_4c':
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c

        elif self._box_rep == 'box_4ca':
            if self._train_val_test in ['train', 'val']:
                prediction_dict[self.PRED_MB_ORIENTATIONS_GT] = \
                    mb_orientations_gt
                prediction_dict[self.PRED_MB_ANGLE_VECTORS] = mb_angle_vectors

            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c
            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orientations

        else:
            raise NotImplementedError('Prediction dict not implemented for',
                                      self._box_rep)

        # prediction_dict[self.PRED_MAX_IOUS] = max_ious
        # prediction_dict[self.PRED_ALL_IOUS] = all_ious

        return prediction_dict


    def sample_mini_batch(self, anchor_box_list_gt, anchor_box_list,
                          class_labels):

        with tf.variable_scope('ammf_create_mb_mask'):
            # Get IoU for every anchor
            all_ious = box_list_ops.iou(anchor_box_list_gt, anchor_box_list)
            max_ious = tf.reduce_max(all_ious, axis=0)
            max_iou_indices = tf.argmax(all_ious, axis=0)

            # Sample a pos/neg mini-batch from anchors with highest IoU match
            mini_batch_utils = self.dataset.kitti_utils.mini_batch_utils
            mb_mask, mb_pos_mask = mini_batch_utils.sample_ammf_mini_batch(
                max_ious)
            mb_class_label_indices = mini_batch_utils.mask_class_label_indices(
                mb_pos_mask, mb_mask, max_iou_indices, class_labels)

            mb_gt_indices = tf.boolean_mask(max_iou_indices, mb_mask)

        return mb_mask, mb_class_label_indices, mb_gt_indices

    def create_feed_dict(self):
        feed_dict = self._p1_model.create_feed_dict()
        self.sample_info = self._p1_model.sample_info
        return feed_dict

    #  only have loss of 3D boxes

    def loss(self, prediction_dict):
        # Note: The loss should be using mini-batch values only
        loss_dict, p1_loss = self._p1_model.loss(prediction_dict)
        losses_output = ammf_loss_builder.build(self, prediction_dict)

        classification_loss = \
            losses_output[ammf_loss_builder.KEY_CLASSIFICATION_LOSS]

        final_reg_loss = losses_output[ammf_loss_builder.KEY_REGRESSION_LOSS]

        ammf_loss = losses_output[ammf_loss_builder.KEY_ammf_LOSS]

        offset_loss_norm = \
            losses_output[ammf_loss_builder.KEY_OFFSET_LOSS_NORM]

        loss_dict.update({self.LOSS_FINAL_CLASSIFICATION: classification_loss})
        loss_dict.update({self.LOSS_FINAL_REGRESSION: final_reg_loss})

        # Add localization and orientation losses to loss dict for plotting
        loss_dict.update({self.LOSS_FINAL_LOCALIZATION: offset_loss_norm})

        ang_loss_loss_norm = losses_output.get(
            ammf_loss_builder.KEY_ANG_LOSS_NORM)
        if ang_loss_loss_norm is not None:
            loss_dict.update({self.LOSS_FINAL_ORIENTATION: ang_loss_loss_norm})

        with tf.variable_scope('model_total_loss'):
            total_loss = p1_loss + ammf_loss

        return loss_dict, total_loss
