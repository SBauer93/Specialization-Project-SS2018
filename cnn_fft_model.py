segment_size = 2048  # length of segments
segment_step = 512  # overlap of segments = segment_size - segment_step

n_classes = 5  # number of classes
fprefix = 'data/07_12_2017_recordings/'  # prefix of recordings
flist = {
        'plastic_spheres_1.wav': 0,
        'plastic_spheres_2.wav': 0,
        'plastic_spheres_3.wav': 0,
        'plastic_spheres_4.wav': 0,
        'ramp_M3_1_steel.wav': 1,
        'ramp_M3_2_steel.wav': 1,
        'ramp_M3_3_steel.wav': 1,
        'ramp_M3_4_steel.wav': 1,
        'ramp_M4_1_steel.wav': 2,
        'ramp_M4_2_steel.wav': 2,
        'ramp_M4_3_steel.wav': 2,
        'ramp_M4_4_steel.wav': 2,
        'ramp_M4_1_messing.wav': 3,
        'ramp_M4_2_messing.wav': 3,
        'ramp_M4_3_messing.wav': 3,
        'ramp_M4_4_messing.wav': 3,
        'ramp_screws_1.wav' : 4,
        'ramp_screws_2.wav' : 4,
        'ramp_screws_3.wav' : 4,
        'ramp_screws_4.wav' : 4
        }


def import_data(fname, nclass, segment_size, segment_step):
    # read wav file
    sr, data = scipy.io.wavfile.read(fname)
    # normalize level
    data = data/np.max(np.abs(data[:]))
    # put both channels into one array
    data = np.ndarray.flatten(data, order='F')
    # segment and sort into feature matrix
    nseg = np.ceil((len(data)-segment_size)/segment_step)
    X = np.array([ data[i*segment_step:i*segment_step+segment_size] for i in range(int(nseg)) ])
    # construct target vector
    # one hot
    y = np.zeros((X.shape[0], n_classes), dtype=np.int)
    y[:, nclass] = 1
    # not one hot
    #np.array(np.ones(X.shape[0])*nclass, dtype=np.int)
    
    return X, y


X = np.empty((0, segment_size))
y = np.empty((0, n_classes), dtype=np.int)
for fname, nclass in flist.items():
    X_t, y_t = import_data(fprefix+fname, nclass, segment_size=segment_size, segment_step=segment_step)
    X = np.append(X, X_t, axis=0)
    y = np.append(y, y_t, axis=0)

#**********************************************************************************************************

print(X.shape)
print(y.shape)

X_f = np.abs(np.fft.rfft(X, axis=1))
print(X_f.shape)
X_f = np.float32(X_f)
# split data into training/test subset
X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.25, random_state=42)
print('loaded {0:5.0f}/{1:<5.0f} training/test samples'.format(X_train.shape[0], X_test.shape[0]))

#**********************************************************************************************************

import tflearn

tf.reset_default_graph()

X_train = np.reshape(X_train, (-1, 1025, 1))
X_test = np.reshape(X_test, (-1, 1025, 1))

# Network building
net = tflearn.input_data([None, 1025, 1])
#net = tflearn.layers.conv.conv_1d(net, 32, 4, activation='relu', regularizer="L2")
#net = tflearn.layers.conv.max_pool_1d(net, 2)
#net = tflearn.layers.normalization.local_response_normalization(net)
net = tflearn.layers.conv.conv_1d(incoming=net, 
                                  nb_filter=40,
                                  filter_size=2,
                                  strides=1,
                                  activation='relu', 
                                  regularizer="L2")
                          
net = tflearn.layers.conv.conv_1d(incoming=net, 
                                  nb_filter=40,
                                  filter_size=2,
                                  strides=1,
                                  activation='relu', 
                                  regularizer="L2")

net = tflearn.layers.conv.max_pool_1d(incoming=net, 
                                      kernel_size=17,
                                      strides=17)
                          
net = tflearn.layers.core.reshape(incoming=net,
                            new_shape=[-1, 61, 40, 1]) 
                          
net = tflearn.layers.conv.conv_2d(incoming=net, 
                                  nb_filter=50,
                                  filter_size=[8, 5],
                                  strides=[1, 1],
                                  activation='relu', 
                                  regularizer="L2")
                          
net = tflearn.layers.conv.max_pool_2d(incoming=net, 
                                      kernel_size=[3, 3],
                                      strides=[3, 3])

net = tflearn.layers.conv.conv_2d(incoming=net, 
                                  nb_filter=50,
                                  filter_size=[1, 4],
                                  strides=[1, 1],
                                  activation='relu', 
                                  regularizer="L2")
                          
net = tflearn.layers.conv.max_pool_2d(incoming=net, 
                                      kernel_size=[1, 1],
                                      strides=[1, 1])

#net = tflearn.layers.normalization.local_response_normalization(net)
net = tflearn.fully_connected(net, 256, activation='relu')
net = tflearn.layers.core.dropout(net, 0.5)
net = tflearn.fully_connected(net, 256, activation='relu')
net = tflearn.layers.core.dropout(net, 0.5)
net = tflearn.fully_connected(net, n_classes, activation='softmax')

# Regression using Adam
#net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
#top_k = tflearn.metrics.Top_k(1)
net = tflearn.regression(net, optimizer=sgd, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X_train, y_train, show_metric=True, n_epoch=100)