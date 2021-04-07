import tensorflow as tf

from ammf.utils.sparse_pool_utils import sparse_pool_layer

slim = tf.contrib.slim


class FusionVggPyr:
    """Define two feature extractors
    """

    def __init__(self,extractor_config_bev,extractor_config_img,M_tf,img_index_flip,bv_index=None):
        # M_tf is the matrix for sparse pooling layer
        self.config_bev = extractor_config_bev
        self.config_img = extractor_config_img
        self.M_tf = M_tf
        self.img_index_flip = img_index_flip
        self.bv_index = bv_index

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    #  build up the vgg pyramid layer with two scources from bev lidar and fv img separately,
    # with the same architecture, fusing feature maps from both sides right at the bottlenecks of backbones


    #def IFM_FeatureFusionModule(input_1, input_2, n_filters):
    def low_fusion(self, input_1, input_2, input_3, n_filters):
   

        input_1 = tf.image.resize_bilinear(input_1, size=[tf.shape(input_3)[1], tf.shape(input_1)[2]])
        input_2 = tf.image.resize_bilinear(input_2, size=[tf.shape(input_3)[1], tf.shape(input_2)[2]])


        inputs = tf.concat((input_1, input_2, input_3), axis=3)
        inputs = slim.conv2d(inputs, n_filters, kernel_size=[3, 3])
        inputs = slim.batch_norm(inputs, fused=True)
        inputs = tf.nn.relu(inputs)

        # Context Modeling, Global average pooling
        net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
        #Transform
        net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
        net = tf.sigmoid(net)

        net = tf.multiply(inputs, net)
        net = tf.add(inputs, net)
        return net

    def high_fusion(self, input_1, input_2, input_3, n_filters):
        
        print(input_1.shape)
        #input_1 = tf.image.resize_bilinear(input_1, size=[min(tf.shape(input_1)[1], tf.shape(input_2)[1]), min(tf.shape(input_1)[2], tf.shape(input_2)[2])]) 
        input_1 = tf.image.resize_bilinear(input_1, size=[tf.shape(input_3)[1], tf.shape(input_1)[2]])
        input_2 = tf.image.resize_bilinear(input_2, size=[tf.shape(input_3)[1], tf.shape(input_2)[2]])

        inputs = tf.concat((input_1, input_2, input_3), axis=3)
        inputs = slim.conv2d(inputs, n_filters, kernel_size=[3, 3])
        inputs = slim.batch_norm(inputs, fused=True)
        inputs = tf.nn.relu(inputs)
        # Context Modeling
        net = slim.conv2d(inputs, n_filters, kernel_size=[1, 1])
        net = tf.nn.softmax(net)
        net = tf.multiply(inputs, net)
        # Transform
        net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
        net = tf.nn.relu(net)
        net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
        net = tf.sigmoid(net)

        net = tf.multiply(inputs, net)
        net = tf.add(inputs, net)
        return net
    
    def multi_fusion(self, input_1, input_2, input_3, input_4, input_5, n_filters):

        input_1 = tf.image.resize_bilinear(input_1, size=[tf.shape(input_2)[1], tf.shape(input_1)[2]])

        inputs24 = tf.concat((input_2, input_4), axis=3)
        print(inputs24.shape)
        inputs24 = tf.image.resize_bilinear(inputs24, size=[tf.shape(inputs24)[1]*2, tf.shape(inputs24)[2]*2])

        inputs15 = tf.concat((input_1, input_5), axis=3)
        
        inputs = tf.concat((inputs15, inputs24), axis=3)

        #inputs = tf.concat([input_1, input_2, input_3, input_4,input_5], axis=3)
        net = slim.conv2d(inputs, n_filters, kernel_size=[1, 1])

        return net

    def build(self,
            inputs_bev,
            inputs_bev_pse,            
            inputs_img,
            input_pixel_size_bev,
            input_pixel_size_img,
            is_training,
            scope_bev = 'bev_vgg_pyr',
            scope_img = 'img_vgg_pyr'):
        

        #  first build two convs
        convs_bev, end_points_bev1 = self._build_individual_before_fusion(inputs_bev,input_pixel_size_bev,is_training,scope_bev,backbone_name='bev')
        convs_bev_pse, end_points_bev1_pse = self._build_individual_before_fusion(inputs_bev_pse,input_pixel_size_bev,is_training,scope_bev,backbone_name='bev')
        convs_img, end_points_img1 = self._build_individual_before_fusion(inputs_img,input_pixel_size_img,is_training,scope_img,backbone_name='img')

        #low fusion
        F_low0= self.low_fusion(convs_bev[0], convs_bev_pse[0], convs_img[0], 32)
        F_low1= self.low_fusion(convs_bev[1], convs_bev_pse[1], convs_img[1], 64)

 
        # do the sparse pooling operation and get the fused conv4_bev and conv4_img
        feature_depths = [self.config_img.vgg_conv4[1],self.config_bev.vgg_conv4[1]] 
        # the depth of the output feature map (for pooled features bev, img respectively)
        # only fusion on the bev_map

        #  then do the deconv to the fused layers
        upconv3_bev, upconv2_bev, upconv1_bev, feature_maps_bev, end_points_bev2 = self._build_individual_after_fusion(convs_bev,input_pixel_size_bev,is_training,scope_bev,backbone_name='bev')
        upconv3_bev_pse, upconv2_bev_pse, upconv1_bev_pse, feature_maps_bev_pse, end_points_bev2_pse = self._build_individual_after_fusion(convs_bev_pse,input_pixel_size_bev,is_training,scope_bev,backbone_name='bev')
        upconv3_img, upconv2_img, upconv1_img, feature_maps_img, end_points_img2 = self._build_individual_after_fusion(convs_img,input_pixel_size_img,is_training,scope_img,backbone_name='img')

        #high fusion  
        F_high3= self.high_fusion(upconv3_bev, upconv3_bev_pse, upconv3_img, 128)
        F_high2= self.high_fusion(upconv2_bev, upconv2_bev_pse, upconv2_img, 64)
        F_high1= self.high_fusion(upconv1_bev, upconv1_bev_pse, upconv1_img, 32)

        #multi fusion
        F_multi = self.multi_fusion(F_low0, F_low1, F_high3, F_high2, F_high1, 64)

        F_multi_bev = tf.image.resize_bilinear(F_multi, size=[tf.shape(feature_maps_bev)[1], tf.shape(feature_maps_bev)[2]])
        feature_maps_bev = tf.concat((F_multi_bev, feature_maps_bev), axis=-1)
        feature_maps_img = tf.concat((F_multi, feature_maps_img), axis=-1)

        # merge the dicts, (for each backbone)
        # requires python3.5 or higher version
        end_points_bev = {**end_points_bev1, **end_points_bev2}
        end_points_img = {**end_points_img1, **end_points_img2}

        return feature_maps_bev, feature_maps_img, end_points_bev, end_points_img

    def _build_individual_before_fusion(self,
              inputs,
              input_pixel_size,
              is_training,
              scope,
              backbone_name):
        """ Modified VGG for BEV feature extraction with pyramid features

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        if backbone_name == 'bev':
            vgg_config = self.config_bev
            scope_mid_name = 'bev_vgg_pyr'
        elif backbone_name == 'img':
            vgg_config = self.config_img
            scope_mid_name = 'img_vgg_pyr'
        else:
            error('Unknown name of single sensor backbone')

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, scope_mid_name, [inputs]) as sc:

                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    if backbone_name == 'bev':
                        # Pad 700 to 704 to allow even divisions for max pooling
                        padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])
                    elif backbone_name == 'img':
                        padded = inputs
                    
                    else:
                        error('Unknown name of single sensor backbone')

                    # Encoder
                    conv1 = slim.repeat(padded,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv1')
                    pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

                    conv2 = slim.repeat(pool1,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')
                    pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

                    conv3 = slim.repeat(pool2,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')
                    pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')

                    conv4 = slim.repeat(pool3,
                                        vgg_config.vgg_conv4[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv4[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)  
                
                return [conv1,conv2,conv3,conv4], end_points

    def _build_individual_after_fusion(self,
              convs,
              input_pixel_size,
              is_training,
              scope,
              backbone_name):    
        if backbone_name == 'bev':
            vgg_config = self.config_bev
            scope_mid_name = 'bev_vgg_pyr'
        elif backbone_name == 'img':
            vgg_config = self.config_img
            scope_mid_name = 'img_vgg_pyr'
        else:
            error('Unknown name of single sensor backbone')

        conv1 = convs[0]
        conv2 = convs[1]
        conv3 = convs[2]
        conv4 = convs[3]
        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, scope_mid_name) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Decoder (upsample and fuse features)
                    # upsample
                    upconv3 = slim.conv2d_transpose(
                        conv4,
                        vgg_config.vgg_conv3[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')
                    
                    # efpn
                    concat3_1 = tf.concat(
                        (conv3, upconv3), axis=3, name='concat3')
                    pyramid_fusion3_1 = slim.conv2d(
                        concat3_1,
                        vgg_config.vgg_conv2[1]/4,
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3_1')

                    pyramid_fusion3_2 = slim.conv2d(
                        pyramid_fusion3_1,
                        vgg_config.vgg_conv2[1]/4,
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3_2')

                    pyramid_fusion3_3 = slim.conv2d(
                        pyramid_fusion3_2,
                        vgg_config.vgg_conv2[1]/4,
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3_3')

                    pyramid_fusion3 = slim.conv2d(
                        pyramid_fusion3_3,
                        vgg_config.vgg_conv2[1]/4,
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3_4')
                    print("eFPN DONE!")

                    upconv2 = slim.conv2d_transpose(
                        pyramid_fusion3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv2')

                    concat2_1 = tf.concat(
                        (conv2, upconv2), axis=3, name='concat2')
                    pyramid_fusion2_1 = slim.conv2d(
                        concat2_1,
                        vgg_config.vgg_conv2[1]/4,
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2_1')

                    pyramid_fusion2_2 = slim.conv2d(
                        pyramid_fusion2_1,
                        vgg_config.vgg_conv2[1]/4,
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2_2')

                    pyramid_fusion2_3 = slim.conv2d(
                        pyramid_fusion2_2,
                        vgg_config.vgg_conv2[1]/4,
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2_3')

                    pyramid_fusion2 = slim.conv2d(
                        pyramid_fusion2_3,
                        vgg_config.vgg_conv2[1]/4,
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2_4')

                    upconv1 = slim.conv2d_transpose(
                        pyramid_fusion_2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv1')

                    concat1_1 = tf.concat(
                        (conv1, upconv1), axis=3, name='concat1')
                    pyramid_fusion1_1 = slim.conv2d(
                        concat1_1,
                        vgg_config.vgg_conv2[1]/4,
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1_1')

                    pyramid_fusion1_2 = slim.conv2d(
                        pyramid_fusion1_1,
                        vgg_config.vgg_conv2[1]/4,
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1_2')

                    pyramid_fusion1_3 = slim.conv2d(
                        pyramid_fusion1_2,
                        vgg_config.vgg_conv2[1]/4,
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1_3')

                    pyramid_fusion1 = slim.conv2d(
                        pyramid_fusion1_3,
                        vgg_config.vgg_conv2[1]/4,
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1_4')

                    # Slice off padded area
                    if backbone_name == 'bev':
                        sliced = pyramid_fusion1[:, 4:]
                    elif backbone_name == 'img':
                        sliced = pyramid_fusion1
                    else:
                        error('Unknown name of single sensor backbone')

                feature_maps_out = sliced

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return upconv3, upconv2, upconv1, feature_maps_out, end_points



    

