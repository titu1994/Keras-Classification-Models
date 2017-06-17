'''
Derived from comment here - https://github.com/tensorflow/serving/issues/310#issuecomment-297015251
'''
import keras.backend as K
from keras.models import Model

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

K.set_learning_phase(0)

'''
Start Editing file here 
'''

# Create new model here and load its weights!
model = None
model.load_weights('')

# Edit the export folder path here !
export_path = ''

'''
No need to edit anything more from here on out
'''

new_model = Model.from_config(model.get_config())
new_model.set_weights(model.get_weights())


builder = saved_model_builder.SavedModelBuilder(export_path)

signature = predict_signature_def(inputs={'images': model.input},
                                  outputs={'scores': model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()


'''
Client side code:

>>> request = predict_pb2.PredictRequest()
>>> request.model_spec.name = '' # <--- update model name here
>>> request.model_spec.signature_name = 'predict'
>>> request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(img))

>>> result = stub.Predict(request, 10.0)  # 10 secs timeout
>>> to_decode = np.expand_dims(result.outputs['outputs'].float_val, axis=0)
>>> decoded = decode_predictions(to_decode, 5)
>>> print(decoded)

'''