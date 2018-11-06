import tensorflow as tf


def get_variable_info(ckpt_file):
  ckpt_reader = tf.train.NewCheckpointReader(ckpt_file)
  shape_map = ckpt_reader.get_variable_to_shape_map()
  type_map = ckpt_reader.get_variable_to_dtype_map()
  return {
    variable_name: (shape_map[variable_name], type_map[variable_name])
    for variable_name in shape_map
  }


if __name__ == "__main__":
    variables = get_variable_info("./resnet_v2_50.ckpt")
    print(variables)
