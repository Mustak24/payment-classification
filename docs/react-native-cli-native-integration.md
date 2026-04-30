# React Native CLI Native Integration (FastText Model)

This guide explains how to integrate the FastText model into a React Native CLI app using native Android and iOS code.

Model files from this repo:
- `models/classifier.ftz` (Quantized model for production)
- `models/metadata.json` (Thresholds and labels info)

Important model constraints:
- Input type: string (whitespace collapsed, no newlines)
- Output: labels (e.g. `__label__CREDIT`) and softmax probabilities
- Confidence threshold: 0.6 (or as defined in `metadata.json`)

---

## 1. Copy model assets into your React Native app

In your React Native project, copy:
- `classifier.ftz`
- `metadata.json`

Recommended destination:
- Android: `android/app/src/main/assets/`
- iOS: Add both files to the Xcode target (Copy Bundle Resources)

---

## 2. Android native setup (Kotlin/C++)

FastText does not have an official pre-built Maven artifact for Android. You will need to either use a community Java wrapper or compile the FastText C++ source code using the Android NDK via JNI.

### 2.1 Add FastText C++ via NDK

1. Add FastText C++ source files to your Android project's `cpp` directory.
2. Create a `CMakeLists.txt` to compile it.
3. Write a JNI wrapper to load the model and call the `predict` function.

### 2.2 Create a native classifier module

Once you have a FastText JNI wrapper (e.g., a custom `FastTextWrapper` class), create a Kotlin module:

- `android/app/src/main/java/<your_package>/TextClassifierModule.kt`

Example implementation:

```kotlin
package com.yourapp

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

class TextClassifierModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    // Hypothetical wrapper around FastText C++ NDK
    private var fastText: FastTextWrapper? = null
    private var threshold: Float = 0.6f

    override fun getName(): String = "TextClassifier"

    init {
        loadMetadata()
        loadModel()
    }

    private fun loadMetadata() {
        val json = reactContext.assets.open("metadata.json").bufferedReader().use { it.readText() }
        val root = JSONObject(json)
        threshold = root.optDouble("confidence_threshold", 0.6).toFloat()
    }

    private fun loadModel() {
        // FastText C++ requires a direct file path.
        // Copy the model from assets to the internal file system first.
        val modelFile = File(reactContext.filesDir, "classifier.ftz")
        if (!modelFile.exists()) {
            reactContext.assets.open("classifier.ftz").use { input ->
                FileOutputStream(modelFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        fastText = FastTextWrapper()
        fastText?.loadModel(modelFile.absolutePath)
    }

    @ReactMethod
    fun predict(text: String, promise: Promise) {
        try {
            val ft = fastText ?: throw IllegalStateException("Model not initialized")

            // Clean text: remove newlines as FastText uses them as record separators
            val cleanedText = text.replace("\n", " ").replace("\r", " ")

            // Call predict (returns top 1 prediction)
            val result = ft.predict(cleanedText, 1)
            
            // Expected result properties: label (String) and probability (Float)
            val rawLabel = result.label
            val confidence = result.probability

            // Strip the FastText label prefix
            val label = rawLabel.replace("__label__", "")

            val finalLabel = if (confidence >= threshold) label else "UNKNOWN"

            val map = com.facebook.react.bridge.Arguments.createMap().apply {
                putString("label", finalLabel)
                putDouble("confidence", confidence.toDouble())
            }
            promise.resolve(map)
        } catch (e: Exception) {
            promise.reject("PREDICT_ERROR", e)
        }
    }
}
```

Also register your module in a package class and add it to `getPackages()`.

---

## 3. iOS native setup (Swift/C++)

FastText is written in C++. You can include the FastText C++ source files directly into your Xcode project and create an Objective-C++ (`.mm`) wrapper to expose it to Swift.

### 3.1 Add FastText to Xcode target

1. Download the FastText C++ source code.
2. Drag the `.cc` and `.h` files into your Xcode project.
3. Create an Objective-C++ wrapper (e.g., `FastTextWrapper.mm` and `FastTextWrapper.h`).

### 3.2 Create Swift bridge module

Example module:

```swift
import Foundation
import React

@objc(TextClassifier)
class TextClassifier: NSObject {
  // Hypothetical Objective-C++ wrapper
  private var fastText: FastTextWrapper?
  private var threshold: Float = 0.6

  override init() {
    super.init()
    loadMetadata()
    loadInterpreter()
  }

  private func loadMetadata() {
    guard let url = Bundle.main.url(forResource: "metadata", withExtension: "json"),
          let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      return
    }

    threshold = (json["confidence_threshold"] as? NSNumber)?.floatValue ?? 0.6
  }

  private func loadInterpreter() {
    // Get absolute path from bundle
    guard let modelPath = Bundle.main.path(forResource: "classifier", ofType: "ftz") else {
      return
    }

    fastText = FastTextWrapper()
    fastText?.loadModel(modelPath)
  }

  @objc
  func predict(_ text: String,
               resolver resolve: @escaping RCTPromiseResolveBlock,
               rejecter reject: @escaping RCTPromiseRejectBlock) {
    do {
      guard let ft = fastText else {
        reject("INIT_ERROR", "Model not initialized", nil)
        return
      }

      // Clean text: remove newlines
      let cleanedText = text.replacingOccurrences(of: "\n", with: " ")
                            .replacingOccurrences(of: "\r", with: " ")

      // Call predict (returns top 1 prediction)
      let prediction = ft.predict(cleanedText, k: 1)
      let labelRaw = prediction.label
      let confidence = prediction.probability
      
      // Strip the FastText label prefix
      let label = labelRaw.replacingOccurrences(of: "__label__", with: "")

      let finalLabel = confidence >= threshold ? label : "UNKNOWN"
      
      resolve([
        "label": finalLabel,
        "confidence": confidence
      ])
    } catch {
      reject("PREDICT_ERROR", "Prediction failed", error)
    }
  }
}
```

Also expose it with Objective-C bridge macros (`RCT_EXTERN_MODULE`).

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

- Model file loads on both platforms (requires copying `.ftz` from assets to device file system on Android).
- `metadata.json` is readable in both platforms.
- Inference works with sample strings.
- Text input is cleaned (newlines removed) before passing to the FastText C++ `predict` function.
- Prediction label strips the `__label__` prefix.
- Prediction label is forced to `UNKNOWN` when confidence < threshold.

---

## 6. Notes specific to this model

- **File Paths:** FastText requires actual file paths to load the model. On Android, assets are compressed inside the APK, so you must copy `classifier.ftz` to the app's internal storage (`filesDir`) before loading it.
- **Newlines:** FastText treats newlines as record separators. Ensure all input strings have `\n` and `\r` replaced with spaces before prediction.
- **Labels:** FastText outputs labels with a prefix (e.g., `__label__CREDIT`). Your native wrapper must strip `__label__` to match the expected app behavior.
