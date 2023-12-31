/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.AccessController.getContext


class ImageClassifierHelper(
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    var currentModel: Int = 4,
    val context: Context,
    val imageClassifierListener: ClassifierListener?
) {
    private var imageClassifier: ImageClassifier? = null

    init {
        setupImageClassifier()
    }

    fun clearImageClassifier() {
        imageClassifier = null
    }


    private fun setupImageClassifier() {
        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    imageClassifierListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
            when (currentModel) {
                CUSTOM_MODEL -> "model.tflite"
                else -> "model.tflite"
            }

        try {
            imageClassifier =
                ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            imageClassifierListener?.onError(
                "Image classifier failed to initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    fun classify(image: Bitmap, rotation: Int) {
        if (imageClassifier == null) {
            setupImageClassifier()
        }

        lateinit var inputImage: TensorImage
        lateinit var outputBuffer: TensorBuffer

        // model 가져오기
        fun copyAssetToFile(context: Context, assetFileName: String): File {
            val outputFile = File(context.filesDir, assetFileName)

            context.assets.open(assetFileName).use { inputStream ->
                FileOutputStream(outputFile).use { outputStream ->
                    val buffer = ByteArray(1024)
                    var length: Int
                    while (inputStream.read(buffer).also { length = it } > 0) {
                        outputStream.write(buffer, 0, length)
                    }
                }
            }
            return outputFile
        }

        // image를 AI에 넣을 수 있게 변형
        fun loadImage(bitmap: Bitmap?): TensorImage{
            // TensorImage에 이미지 데이터 저장
            inputImage.load(bitmap)
            // 전처리 ImageProcessor 정의
            val imageProcessor =
                ImageProcessor.Builder()                            // Builder 생성
                    .add(ResizeOp(224,224, // 이미지 크기 변환
                        ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .add(NormalizeOp(0.0f, 255.0f))                 // 이미지 정규화
                    // .add(CastOp(DataType.UINT8))
                    .build()                                        // ImageProcessor 생성
            // 이미지를 전처리하여 TensorImage 형태로 반환
            return imageProcessor.process(inputImage)
        }

        // 모델 생성
        val interpreter :Interpreter = Interpreter(copyAssetToFile(context, "model2.tflite"))

        // 인풋 이미지 전처리
        val inputTensor = interpreter.getInputTensor(0)
        inputImage = TensorImage(inputTensor.dataType())
        val tensorInputImage : TensorImage = loadImage(image)

        // 아웃풋 할당
        val outputTensor = interpreter.getOutputTensor(0)
        outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

        // 모델 추론
        interpreter?.run(tensorInputImage.buffer, outputBuffer.buffer.rewind())

        // 최종 인덱스(0: 비상구, 1: 소화기, 3: 소화전)
        val predicted_index : Int = outputBuffer.floatArray.indices.maxByOrNull { outputBuffer.floatArray[it] } ?: -1

        println(predicted_index)

    }

    // Receive the device rotation (Surface.x values range from 0->3) and return EXIF orientation
    // http://jpegclub.org/exif_orientation.html
    private fun getOrientationFromRotation(rotation: Int) : ImageProcessingOptions.Orientation {
        when (rotation) {
            Surface.ROTATION_270 ->
                return ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_180 ->
                return ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            Surface.ROTATION_90 ->
                return ImageProcessingOptions.Orientation.TOP_LEFT
            else ->
                return ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(
            results: List<Classifications>?,
            inferenceTime: Long
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTNETV0 = 1
        const val MODEL_EFFICIENTNETV1 = 2
        const val MODEL_EFFICIENTNETV2 = 3
        const val CUSTOM_MODEL = 4

        private const val TAG = "ImageClassifierHelper"
    }
}
