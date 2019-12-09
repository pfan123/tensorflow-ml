const tf = require('@tensorflow/tfjs')
const fs = require('fs')
// const getPixels = require("get-pixels")
const cv = require('opencv4nodejs')
const { map, forEach, startsWith, shuffle } = require('lodash')
// const Blob = require('blob')

const ImageWidth = 32
const ImageHeight = 32
const DIR_PATH = '/Users/linyi/Downloads/视觉风格大纲/'


function loadImgFromDir(dir_path) {
  const dirs = fs.readdirSync(dir_path)
  const datas = []
  forEach(dirs, dirName => {
    const dir = fs.lstatSync(dir_path+dirName)
    if (!dir.isDirectory()) return // 非目录的文件跳过
    const files = fs.readdirSync(dir_path+dirName)
    
    forEach(files, fileName => {
      if (startsWith(fileName, '.')) return // 如果以 .开头则跳过文件
      const path = dir_path+dirName+'/'+fileName
      const data = fs.readFileSync(path)
      const image = cv.imread(path)
      const res = image.resize(ImageWidth, ImageHeight)
      datas.push({ image: res.getDataAsArray(), label: dirName })
    })
  })
  return datas
}

async function loadData() {
  const datas = loadImgFromDir(DIR_PATH)
  
  // 打乱顺序
  const shuffleDatas = shuffle(datas)
  const threshold = 0.75 * shuffleDatas.length
  const imageShape = [ImageWidth, ImageHeight, 3]
  const trainImages = []
  const trainLabels = []
  const testImages = []
  const testLabels = []
  // console.log('test: ', shuffleDatas[0])
  forEach(shuffleDatas, (data, index) => {
    if (index < threshold) {
      // 小于阈值的都认为训练集
      trainImages.push(data.image)
      trainLabels.push(data.label)
    } else {
      // 大于阈值的都认为测试集
      testImages.push(data.image)
      testLabels.push(data.label)
    }
  })

  console.log('test: ', testLabels[0])

  console.log('datas: ', shuffleDatas.length, ', threshold:', threshold)
  // console.log('trainData:', trainImages[10], ', label:', trainLabels[10])
  let b = new Buffer(shuffleDatas)
  console.log('memory:', )
  return { 
    trainImages: tf.tensor4d(trainImages), 
    trainLabels: tf.oneHot(tf.tensor1d(trainLabels, 'int32'), 8).toFloat(), 
    testImages: tf.tensor4d(testImages), 
    testLabels: tf.oneHot(tf.tensor1d(testLabels, 'int32'), 8).toFloat(), 
  }
  // console.log('res', res)
}

function imageByteArray (image, numChannels) {
  const pixels = image.data
  const numPixels = image.width * image.height;
  const values = new Int32Array(numPixels * numChannels);

  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numChannels; ++channel) {
      values[i * numChannels + channel] = pixels[i * 4 + channel];
    }
  }

  return values
}

module.exports = {
  loadData
}