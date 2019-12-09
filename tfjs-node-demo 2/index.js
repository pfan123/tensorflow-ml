const Koa = require('koa')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const { loadData } = require('./data')
const staticModel = require('./model')
const Router = require('koa-router')
const router = Router()
const app = new Koa()
const Apisauce = require('apisauce')
const cv = require('opencv4nodejs')

const LEARNING_RATE = 0.001

async function start () {
  console.log('for test')
  let model = await tf.loadModel('file://model/model.json')

  if (!model) {
    model = staticModel
  }

  const optimizer = tf.train.adam(LEARNING_RATE)
  model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })
  const { trainImages, trainLabels, testImages, testLabels } = await loadData()
  await model.fit(trainImages, trainLabels, {
    epochs: 50,
    batchSize: 128
  })
  console.log('evluate Test...')
  const evalOutput = model.evaluate(testImages, testLabels)
  console.log(
    '\nEvaluation result:\n' +
    `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
    `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`)
  await model.save('file://model')
  // console.log('res:', res)
}

router.get('/predict', async ctx => {
  console.log('predict')
  const model = await tf.loadModel('file://model/model.json')
  const optimizer = tf.train.adam(LEARNING_RATE)

  model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })
  const api = Apisauce.create({ baseURL: 'https://img20.360buyimg.com/ling/s200x0_jfs/t10906/103/182552109/164752/e1ebef5e/59ca12c5Nd9416ae6.png' })
  const { ok, data } = await api.get('')
  if (ok && data) {
    // console.log('imageData:', data)
    const img = cv.imdecode(data)
    console.log('img:', img)
  }
  // const img = cv.imread('https://img20.360buyimg.com/ling/s200x0_jfs/t10906/103/182552109/164752/e1ebef5e/59ca12c5Nd9416ae6.png')
  // console.log('img: ', img)
  // model.predict()
})

router.get('/', ctx => {
  start()
  ctx.response.body = 'Hello Koa'
})

app.use(router.routes())
app.listen(3311)
console.log(' server start at port 3311 ')
