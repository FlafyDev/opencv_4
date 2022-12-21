package ar.fgsoruco.opencv4

import java.io.*
import java.nio.file.*
import java.nio.file.Paths

import androidx.annotation.NonNull
import ar.fgsoruco.opencv4.factory.colormaps.ApplyColorMapFactory
import ar.fgsoruco.opencv4.factory.colorspace.CvtColorFactory
import ar.fgsoruco.opencv4.factory.imagefilter.*
import ar.fgsoruco.opencv4.factory.miscellaneous.AdaptiveThresholdFactory
import ar.fgsoruco.opencv4.factory.miscellaneous.DistanceTransformFactory
import ar.fgsoruco.opencv4.factory.miscellaneous.ThresholdFactory

import io.flutter.FlutterInjector;
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import io.flutter.plugin.common.PluginRegistry.Registrar
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs

import io.flutter.plugin.common.StandardMethodCodec


/** Opencv_4Plugin */
class Opencv4Plugin: FlutterPlugin, MethodCallHandler {
  var OpenCVFLag = false
  /// The MethodChannel that will the communication between Flutter and native Android
  ///
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it
  /// when the Flutter Engine is detached from the Activity
  private lateinit var channel : MethodChannel
  private lateinit var hogDescriptorFace: HOGDescriptor
  private lateinit var cascadeClassifier: CascadeClassifier
  private var binding: FlutterPlugin.FlutterPluginBinding? = null

  override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    val taskQueue =
      flutterPluginBinding.binaryMessenger.makeBackgroundTaskQueue()
    channel = MethodChannel(flutterPluginBinding.binaryMessenger,
                            "opencv_4",
                            StandardMethodCodec.INSTANCE,
                            taskQueue)
    channel.setMethodCallHandler(this)
    binding = flutterPluginBinding;
  }

  override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
    if (!OpenCVFLag) {
      if (!OpenCVLoader.initDebug()) {
        println("Error on load OpenCV")
      } else {
        OpenCVFLag = true;
      }
      hogDescriptorFace = HOGDescriptor();
      hogDescriptorFace.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());

      val outputDir = binding!!.getApplicationContext().getCacheDir();
      val file: File = File.createTempFile("temp", ".xml", outputDir);

      val assetPath: String = binding!!
        .getFlutterAssets()
        .getAssetFilePathBySubpath("assets/haarcascade_frontalface_default.xml", "opencv_4");

      val inputStream = binding!!.getApplicationContext().getAssets().open(assetPath);
      inputStream.use { input ->
        Files.copy(input, file.toPath(), StandardCopyOption.REPLACE_EXISTING)
      }

      cascadeClassifier = CascadeClassifier(file.absolutePath);
    }

    when (call.method) {
      "containsFace" -> {
        try {
          val data = call.argument<ByteArray>("data") as ByteArray;
          val src = Imgcodecs.imdecode(MatOfByte(*data), Imgcodecs.IMREAD_UNCHANGED);

          val faces = MatOfPoint();
          val weights = MatOfDouble();
  
          hogDescriptorFace.detect(src, faces, weights, 0.0, Size(2.0, 2.0), Size(8.0, 8.0))
              // Size(2.0, 2.0), //winSize
              // Size(0.0, 0.0), //blocksize
              // 1.05,
              // 2.0,false);

          if (faces.toList().size > 0) {
            result.success(true);
            return;
            // weights.toArray().forEach { 
            //   if (it > 0.2) {
            //     result.success(true);
            //     return;
            //   }
            // }
          }

          val faces2 = MatOfRect();
          cascadeClassifier.detectMultiScale(src, faces2, 1.1);

          if (faces2.toList().size > 0) {
            result.success(true);
            return;
          }

          result.success(false)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }

      "getVersion" -> {
        try {
          result.success(Core.VERSION)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }

      //Module: Image Filtering
      "bilateralFilter" -> {
        try {
          BilateralFilterFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("diameter") as Int,
                  call.argument<Int>("sigmaColor") as Int,
                  call.argument<Int>("sigmaSpace") as Int,
                  call.argument<Int>("borderType") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "blur" -> {
        try {
          BlurFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<ArrayList<Double>>("kernelSize") as ArrayList<Double>,
                  call.argument<ArrayList<Double>>("anchorPoint") as ArrayList<Double>,
                  call.argument<Int>("borderType") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "boxFilter" -> {
        try {
          BoxFilterFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("outputDepth") as Int,
                  call.argument<ArrayList<Double>>("kernelSize") as ArrayList<Double>,
                  call.argument<ArrayList<Double>>("anchorPoint") as ArrayList<Double>,
                  call.argument<Boolean>("normalize") as Boolean,
                  call.argument<Int>("borderType") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "dilate" -> {
        try {
          DilateFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<ArrayList<Double>>("kernelSize") as ArrayList<Double>,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "erode" -> {
        try {
          ErodeFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<ArrayList<Double>>("kernelSize") as ArrayList<Double>,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "filter2D" -> {
        try {
          Filter2DFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("outputDepth") as Int,
                  call.argument<ArrayList<Int>>("kernelSize") as ArrayList<Int>,
                  result
          )
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "gaussianBlur" -> {
        try {
          GaussianBlurFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<ArrayList<Double>>("kernelSize") as ArrayList<Double>,
                  call.argument<Double>("sigmaX") as Double,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "laplacian" -> {
        try {
          LaplacianFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("depth") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "medianBlur" -> {
        try {
          MedianBlurFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("kernelSize") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "morphologyEx" -> {
        try {
          MorphologyExFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("operation") as Int,
                  call.argument<ArrayList<Int>>("kernelSize") as ArrayList<Int>,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "pyrMeanShiftFiltering" -> {
        try {
          PyrMeanShiftFilteringFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Double>("spatialWindowRadius") as Double,
                  call.argument<Double>("colorWindowRadius") as Double,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "scharr" -> {
        try {
          ScharrFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("depth") as Int,
                  call.argument<Int>("dx") as Int,
                  call.argument<Int>("dy") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "sobel" -> {
        try {
          SobelFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("depth") as Int,
                  call.argument<Int>("dx") as Int,
                  call.argument<Int>("dy") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "sqrBoxFilter" -> {
        try {
          SqrBoxFilterFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("outputDepth") as Int,
                  call.argument<ArrayList<Double>>("kernelSize") as ArrayList<Double>,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      //Module: Color Maps
      "applyColorMap" -> {
        try {
          //colorMap: Int
          ApplyColorMapFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("colorMap") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: " + e.message, e)
        }
      }
      //Module: Color Space Conversions
      "cvtColor" -> {
        try {
          CvtColorFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("outputType") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      //Module: Miscellaneous Image Transformations
      "adaptiveThreshold" -> {
        try {
          AdaptiveThresholdFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Double>("maxValue") as Double,
                  call.argument<Int>("adaptiveMethod") as Int,
                  call.argument<Int>("thresholdType") as Int,
                  call.argument<Int>("blockSize") as Int,
                  call.argument<Double>("constantValue") as Double,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "distanceTransform" -> {
        //distanceType: Int, maskSize: Int
        try {
          DistanceTransformFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Int>("distanceType") as Int,
                  call.argument<Int>("maskSize") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }
      "threshold" -> {
        try {
          ThresholdFactory.process(
                  call.argument<Int>("pathType") as Int,
                  call.argument<String>("pathString") as String,
                  call.argument<ByteArray>("data") as ByteArray,
                  call.argument<Double>("thresholdValue") as Double,
                  call.argument<Double>("maxThresholdValue") as Double,
                  call.argument<Int>("thresholdType") as Int,
                  result)
        } catch (e: Exception) {
          result.error("OpenCV-Error", "Android: "+e.message, e)
        }
      }

      else -> result.notImplemented()
    }
  }

  override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
    this.binding = null;
  }
}
