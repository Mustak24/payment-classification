# React Native CLI Native Integration (TFLite Model)

This guide explains how to integrate this model into a React Native CLI app using native Android and iOS code.

Model files from this repo:
- `models/classifier.tflite`
- `models/tflite_labels.json`

Important model constraints:
- Input type: string
- Output: softmax probabilities for classes `CREDIT`, `DEBIT`, `UNKNOWN`
- Confidence threshold: 0.6
- Requires Select TF Ops: true

---

## 1. Copy model assets into your React Native app

In your React Native project, copy:
- `classifier.tflite`
- `tflite_labels.json`

Recommended destination:
- Android: `android/app/src/main/assets/`
- iOS: Add both files to the Xcode target (Copy Bundle Resources)

---

## 2. Android native setup (Kotlin)

### 2.1 Add TensorFlow Lite dependencies

In `android/app/build.gradle` (or module-level Gradle file), add:

```gradle
dependencies {
    implementation "org.tensorflow:tensorflow-lite:2.14.0"
    implementation "org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0"
}
```

If your app does not already package native libs correctly, add:

```gradle
android {
    packagingOptions {
        pickFirst "**/*.so"
    }
}
```

### 2.2 Create a native classifier module

Create a Kotlin module, for example:
- `android/app/src/main/java/<your_package>/TextClassifierModule.kt`

Example implementation:

```kotlin
package com.yourapp

import android.content.res.AssetFileDescriptor
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TextClassifierModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private var interpreter: Interpreter? = null
    private var classes: List<String> = emptyList()
    private var threshold: Float = 0.6f

    override fun getName(): String = "TextClassifier"

    init {
        loadMetadata()
        loadModel()
    }

    private fun loadMetadata() {
        val json = reactContext.assets.open("tflite_labels.json").bufferedReader().use { it.readText() }
        val root = JSONObject(json)
        val arr = root.getJSONArray("classes")
        classes = (0 until arr.length()).map { arr.getString(it) }
        threshold = root.optDouble("confidence_threshold", 0.6).toFloat()
    }

    private fun loadModel() {
        val modelBuffer = loadModelFile("classifier.tflite")
        interpreter = Interpreter(modelBuffer)
    }

    private fun loadModelFile(assetName: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = reactContext.assets.openFd(assetName)
        FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
            val fileChannel = inputStream.channel
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )
        }
    }

    @ReactMethod
    fun predict(text: String, promise: Promise) {
        try {
            val localInterpreter = interpreter ?: throw IllegalStateException("Interpreter not initialized")

            // Model input is a single string tensor with shape [1]
            val input = arrayOf(text)

            // Output shape is [1, numClasses]
            val output = Array(1) { FloatArray(classes.size) }

            localInterpreter.run(input, output)
            val probs = output[0]

            var bestIdx = 0
            var bestScore = probs[0]
            for (i in 1 until probs.size) {
                if (probs[i] > bestScore) {
                    bestScore = probs[i]
                    bestIdx = i
                }
            }

            val predicted = if (bestScore >= threshold) classes[bestIdx] else "UNKNOWN"

            val result = com.facebook.react.bridge.Arguments.createMap().apply {
                putString("label", predicted)
                putDouble("confidence", bestScore.toDouble())
            }
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("PREDICT_ERROR", e)
        }
    }
}
```

Also register your module in a package class and add it to `getPackages()`.

---

## 3. iOS native setup (Swift)

### 3.1 Add TensorFlow Lite iOS libraries

If you use CocoaPods, in `ios/Podfile` add:

```ruby
pod 'TensorFlowLiteSwift'
pod 'TensorFlowLiteSelectTfOps'
```

Then run in the iOS folder:

```bash
pod install
```

### 3.2 Add model files to Xcode target

Drag:
- `classifier.tflite`
- `tflite_labels.json`

into Xcode project and make sure they are included in your app target.

### 3.3 Create Swift bridge module

Example module:

```swift
import Foundation
import TensorFlowLite

@objc(TextClassifier)
class TextClassifier: NSObject {
  private var interpreter: Interpreter?
  private var classes: [String] = []
  private var threshold: Float = 0.6

  override init() {
    super.init()
    loadMetadata()
    loadInterpreter()
  }

  private func loadMetadata() {
    guard let url = Bundle.main.url(forResource: "tflite_labels", withExtension: "json"),
          let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      return
    }

    classes = (json["classes"] as? [String]) ?? []
    threshold = (json["confidence_threshold"] as? NSNumber)?.floatValue ?? 0.6
  }

  private func loadInterpreter() {
    guard let modelPath = Bundle.main.path(forResource: "classifier", ofType: "tflite") else {
      return
    }

    do {
      let options = Interpreter.Options()
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      try interpreter?.allocateTensors()
    } catch {
      print("Interpreter init failed: \(error)")
    }
  }

  @objc
  func predict(_ text: String,
               resolver resolve: @escaping RCTPromiseResolveBlock,
               rejecter reject: @escaping RCTPromiseRejectBlock) {
    do {
      guard let interpreter = interpreter else {
        reject("INIT_ERROR", "Interpreter not initialized", nil)
        return
      }

      // For string input tensor, pass UTF-8 bytes to input tensor 0.
      // Depending on TensorFlowLite version, string tensor helper APIs may differ.
      let data = text.data(using: .utf8) ?? Data()
      try interpreter.copy(data, toInputAt: 0)
      try interpreter.invoke()

      let outputTensor = try interpreter.output(at: 0)
      let probabilities = outputTensor.data.withUnsafeBytes {
        Array($0.bindMemory(to: Float32.self))
      }

      guard !probabilities.isEmpty, !classes.isEmpty else {
        reject("OUTPUT_ERROR", "Invalid model output", nil)
        return
      }

      var bestIdx = 0
      var bestScore = probabilities[0]
      for i in 1..<probabilities.count {
        if probabilities[i] > bestScore {
          bestScore = probabilities[i]
          bestIdx = i
        }
      }

      let label = bestScore >= threshold ? classes[bestIdx] : "UNKNOWN"
      resolve([
        "label": label,
        "confidence": bestScore
      ])
    } catch {
      reject("PREDICT_ERROR", "Prediction failed", error)
    }
  }
}
```

Also expose it with Objective-C bridge macros (`RCT_EXTERN_MODULE`) or directly in Objective-C if your app uses an Obj-C bridge.

---

## 4. JavaScript usage from React Native

Example JS call:

```ts
import { NativeModules } from 'react-native';

const { TextClassifier } = NativeModules;

export async function classifyText(text: string) {
  const result = await TextClassifier.predict(text);
  // result: { label: 'CREDIT' | 'DEBIT' | 'UNKNOWN', confidence: number }
  return result;
}
```

---

## 5. Validation checklist

- Model file loads on both platforms.
- `tflite_labels.json` is readable in both platforms.
- Inference works with sample strings.
- Prediction label is forced to `UNKNOWN` when confidence < 0.6.
- Android includes `tensorflow-lite-select-tf-ops`.
- iOS includes `TensorFlowLiteSelectTfOps`.

---

## 6. Notes specific to this model

- This model has a built-in `TextVectorization` layer and expects raw text string input.
- The conversion was done with Select TF Ops enabled; do not remove Select TF Ops dependencies.
- If you retrain and export a new model, always ship the matching `tflite_labels.json` together with `classifier.tflite`.
