import tensorflow as tf
from tensorflow.python.framework import tensor_util
import os
import shutil

def convert_pb_to_saved_model(pb_model_path, export_dir):
    # 1. Load the frozen GraphDef from the .pb file
    with tf.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # 2. Create a simple wrapper that uses the frozen graph
    def import_fn():
        tf.compat.v1.import_graph_def(graph_def, name="")
        return tf.compat.v1.get_default_graph()

    # 3. Use wrap_function to create a concrete function
    # The key is to NOT provide input signatures to wrap_function, but instead 
    # define them when calling the concrete function
    concrete_func = tf.compat.v1.wrap_function(
        lambda: import_fn().get_tensor_by_name("Identity:0"),
        []
    )
    
    # 4. Create a SignatureDef manually and save using low-level API
    print(f"Exporting frozen graph to SavedModel at: {export_dir}")
    
    # Clean up if directory exists
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    
    # Use tf.saved_model with explicit signature
    from tensorflow.python.saved_model import builder as saved_model_builder
    from tensorflow.python.saved_model import signature_def_utils
    from tensorflow.python.saved_model import utils as saved_model_utils
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.saved_model import signature_constants
    
    builder = saved_model_builder.SavedModelBuilder(export_dir)
    
    # Create a graph for serving
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name="")
        
        # Get tensor references
        input_ids_tensor = graph.get_tensor_by_name("input_ids:0")
        attention_mask_tensor = graph.get_tensor_by_name("attention_mask:0")
        token_type_ids_tensor = graph.get_tensor_by_name("token_type_ids:0")
        output_tensor = graph.get_tensor_by_name("Identity:0")
        
        # Create signature
        signature_inputs = {
            "input_ids": saved_model_utils.build_tensor_info(input_ids_tensor),
            "attention_mask": saved_model_utils.build_tensor_info(attention_mask_tensor),
            "token_type_ids": saved_model_utils.build_tensor_info(token_type_ids_tensor)
        }
        signature_outputs = {
            "output": saved_model_utils.build_tensor_info(output_tensor)
        }
        
        signature_def = signature_def_utils.build_signature_def(
            signature_inputs,
            signature_outputs,
            signature_constants.PREDICT_METHOD_NAME
        )
        
        with tf.compat.v1.Session(graph=graph) as session:
            builder.add_meta_graph_and_variables(
                session,
                [tag_constants.SERVING],
                signature_def_map={"serving_default": signature_def}
            )
    
    builder.save()


# Execute the conversion
# Adjust paths to match your current directory structure
convert_pb_to_saved_model(
    pb_model_path="isidorus_bert/bert_model.pb", 
    export_dir="bert_model.savedmodel/"
)
print("Conversion completed successfully!")