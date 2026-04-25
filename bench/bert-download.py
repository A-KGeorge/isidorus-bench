import os
import sys

# TRICK: Force transformers to use tf_keras instead of the new Keras 3
try:
    import tf_keras
    sys.modules['keras'] = tf_keras
except ImportError:
    print("Error: tf-keras not found. Please run: pip install tf-keras")
    sys.exit(1)

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Now import transformers
from transformers import TFBertModel

# 1. Load Model
print("Downloading/Loading bert-base-uncased...")
model = TFBertModel.from_pretrained("bert-base-uncased")

# 2. Define the Inference Function
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 128], dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=[1, 128], dtype=tf.int32, name="attention_mask"),
    tf.TensorSpec(shape=[1, 128], dtype=tf.int32, name="token_type_ids")
])
def serving_fn(input_ids, attention_mask, token_type_ids):
    outputs = model(input_ids, attention_mask, token_type_ids, training=False)
    return outputs[0] 

# 3. Trace and Freeze
print("Freezing graph to .pb...")
concrete_func = serving_fn.get_concrete_function()
frozen_func = convert_variables_to_constants_v2(concrete_func)

# 4. Save to .pb
output_dir = "./isidorus_bert"
os.makedirs(output_dir, exist_ok=True)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=output_dir,
                  name="bert_model.pb",
                  as_text=False)

print(f"\n--- SUCCESS ---")
print(f"File: {output_dir}/bert_model.pb")
print(f"Input nodes: {[t.name.split(':')[0] for t in frozen_func.inputs]}")
print(f"Output nodes: {[t.name.split(':')[0] for t in frozen_func.outputs]}")