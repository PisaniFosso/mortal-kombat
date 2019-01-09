import * as tf from '@tensorflow/tfjs';

const model = tf.sequential();
model.add(tf.layers.inputLayer({ inputShape: [1024] }));
model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
model.compile({
  optimizer: tf.train.adam(1e-6),
  loss: tf.losses.sigmoidCrossEntropy,
  metrics: ['accuracy']
});

export const loadModel = async () => {
  const mn = new mobilenet.MobileNet(1, 1);
  mn.path = `mobile-net/model.json`;
  await mn.load();
  return (input): tf.Tensor1D =>
      mn.infer(input, 'global_average_pooling2d_1')
        .reshape([1024]);
};

const punches = require('videos/Dataset/Kicks-aug')
  .readdirSync(Punches)
  .filter(f => f.endsWith('.jpg'))
  .map(f => `${Punches}/${f}`);

const others = require('videos/Dataset/Others-aug')
  .readdirSync(Others)
  .filter(f => f.endsWith('.jpg'))
  .map(f => `${Others}/${f}`);

const ys = tf.tensor1d(
  new Array(punches.length).fill(1)
    .concat(new Array(others.length).fill(0)));

const xs: tf.Tensor2D = tf.stack(
  punches
    .map((path: string) => mobileNet(readInput(path)))
    .concat(others.map((path: string) => mobileNet(readInput(path))))
 as tf.Tensor2D;

 export const readInput = img => imageToInput(readImage(img), TotalChannels);

const readImage = path => jpeg.decode(fs.readFileSync(path), true);

const imageToInput = image => {
  const values = serializeImage(image);
  return tf.tensor3d(values, [image.height, image.width, 3], 'int32');
};

const serializeImage = image => {
  const totalPixels = image.width * image.height;
  const result = new Int32Array(totalPixels * 3);
  for (let i = 0; i < totalPixels; i++) {
    result[i * 3 + 0] = image.data[i * 4 + 0];
    result[i * 3 + 1] = image.data[i * 4 + 1];
    result[i * 3 + 2] = image.data[i * 4 + 2];
  }
  return result;
}


await model.fit(xs, ys, {
  epochs: Epochs,
  batchSize: parseInt(((punches.length + others.length) * BatchSize).toFixed(0)),
  callbacks: {
    onBatchEnd: async (_, logs) => {
      console.log('Cost: %s, accuracy: %s', logs.loss.toFixed(5), logs.acc.toFixed(5));
      await tf.nextFrame();
    }
  }
});