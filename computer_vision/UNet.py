import tensorflow as tf

from tensorflow.keras import backend
backend.set_image_data_format('channels_first')


# input layer
input_layer = tf.keras.layers.Input(shape=(4, 160, 160, 16)) # (channels, h, w, layer)

# contraction (downward path)
#
# depth 0
down_convl_depth0_layer0 = tf.keras.layers.Conv3D(32,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',                                                  
                                                  activation='relu')(input_layer)

down_convl_depth0_layer1 = tf.keras.layers.Conv3D(64,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_convl_depth0_layer0)


down_mpool_depth0 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(down_convl_depth0_layer1)

# depth 1
down_convl_depth1_layer0 = tf.keras.layers.Conv3D(64,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_mpool_depth0)

down_convl_depth1_layer1 = tf.keras.layers.Conv3D(128,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_convl_depth1_layer0)

down_mpool_depth1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(down_convl_depth1_layer1)

# depth 2
down_convl_depth2_layer0 = tf.keras.layers.Conv3D(128,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_mpool_depth1)

down_convl_depth2_layer1 = tf.keras.layers.Conv3D(256,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_convl_depth2_layer0)

down_mpool_depth2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(down_convl_depth2_layer1)

# depth 3
down_convl_depth3_layer0 = tf.keras.layers.Conv3D(256,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_mpool_depth2)

down_convl_depth3_layer1 = tf.keras.layers.Conv3D(512,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_convl_depth3_layer0)

down_mpool_depth3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(down_convl_depth3_layer1)

# depth 4
down_convl_depth4_layer0 = tf.keras.layers.Conv3D(512,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_mpool_depth3)

down_convl_depth4_layer1 = tf.keras.layers.Conv3D(1024,
                                                  kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1),
                                                  padding='same',
                                                  activation='relu')(down_convl_depth4_layer0)

up_sampling_depth4 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(down_convl_depth4_layer1)

# expansion (upward path)
#
# depth 3
up_concat_depth3 = tf.keras.layers.concatenate([up_sampling_depth4, down_convl_depth3_layer1], axis=1)

up_convl_depth3_layer0 = tf.keras.layers.Conv3D(256,
                                                kernel_size=(3, 3, 3),
                                                strides=(1, 1, 1),
                                                padding='same',
                                                activation='relu')(up_concat_depth3)

up_convl_depth3_layer1 = tf.keras.layers.Conv3D(512,
                                                kernel_size=(3, 3, 3),
                                                strides=(1, 1, 1),
                                                padding='same',
                                                activation='relu')(up_convl_depth3_layer0)

up_sampling_depth3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(up_convl_depth3_layer1)

# depth 2
up_concat_depth2 = tf.keras.layers.concatenate([up_sampling_depth3, down_convl_depth2_layer1], axis=1)

up_convl_depth2_layer0 = tf.keras.layers.Conv3D(128,
                                                kernel_size=(3, 3, 3),
                                                strides=(1, 1, 1),
                                                padding='same',
                                                activation='relu')(up_concat_depth2)

up_convl_depth2_layer1 = tf.keras.layers.Conv3D(256,
                                                kernel_size=(3, 3, 3),
                                                strides=(1, 1, 1),
                                                padding='same',
                                                activation='relu')(up_convl_depth2_layer0)

up_sampling_depth2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(up_convl_depth2_layer1)

# depth 1
up_concat_depth1 = tf.keras.layers.concatenate([up_sampling_depth2, down_convl_depth1_layer1], axis=1)

up_convl_depth1_layer0 = tf.keras.layers.Conv3D(64,
                                                kernel_size=(3, 3, 3),
                                                strides=(1, 1, 1),
                                                padding='same',
                                                activation='relu')(up_concat_depth1)

up_convl_depth1_layer1 = tf.keras.layers.Conv3D(128,
                                                kernel_size=(3, 3, 3),
                                                strides=(1, 1, 1),
                                                padding='same',
                                                activation='relu')(up_convl_depth1_layer0)

up_sampling_depth1 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(up_convl_depth1_layer1)

# depth 1
up_concat_depth0 = tf.keras.layers.concatenate([up_sampling_depth1, down_convl_depth0_layer1], axis=1)

up_convl_depth0_layer0 = tf.keras.layers.Conv3D(32,
                                                kernel_size=(3, 3, 3),
                                                strides=(1, 1, 1),
                                                padding='same',
                                                activation='relu')(up_concat_depth0)

up_convl_depth0_layer1 = tf.keras.layers.Conv3D(64,
                                                kernel_size=(3, 3, 3),
                                                strides=(1, 1, 1),
                                                padding='same',
                                                activation='relu')(up_convl_depth0_layer0)

# output layer
final_conv = tf.keras.layers.Conv3D(3,
                                    kernel_size=(1, 1, 1),
                                    strides=(1, 1, 1),
                                    padding='valid',
                                    activation='sigmoid')(up_convl_depth0_layer1)

# final model
u_net_model = tf.keras.models.Model(inputs=input_layer, outputs=final_conv)
print(u_net_model.summary())