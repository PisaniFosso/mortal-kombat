export const loadModel = async () => {
  const mn = new mobilenet.MobileNet(1, 1);
  mn.path = `C:/Users/Pisani Fosso/Documents/Youcam/Mortal kombat/mobile-net/model.json`;
  await mn.load();
  return (input): tf.Tensor1D =>
      mn.infer(input, 'global_average_pooling2d_1')
        .reshape([1024]);
}; 