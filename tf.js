// import * as tf from '@tensorflow/tfjs';
// const tf = require('@tensorflow/tfjs-node-gpu')
const tf = require('@tensorflow/tfjs-node')

// tensorflow 基本构建块 张量 变量 操作
// Define a model for linear regression.
const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

const shape = [2, 3]
// shape.length === 2 shape[0]行，shape[1] 列 shape.length === 1 shape[0] 列 1 行
// const a = tf.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
const a = tf.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6])
a.dispose()

// 内存管理 dispose tidy
tf.tidy()
a.print()

// const c = tf.tensor2d([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], [2, 3])
// c.print()

// const zero = tf.zeros([4, 3])
// zero.print()

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1])
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1])

// const
const s = tf.scalar(11)
s.print()

// variable
const initialValues = tf.zeros([4])
const biases = tf.variable(initialValues)
biases.print()

const updateValues = tf.tensor1d([0, 1, 0, 1])
biases.assign(updateValues)
biases.print()

// Train the model using the data.
model.fit(xs, ys).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print()
})
