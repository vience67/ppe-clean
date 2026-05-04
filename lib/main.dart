import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
import 'package:flutter/services.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'PPE Detector',
    theme: ThemeData.dark(),
    home: const PPECameraScreen(),
  );
}

class PPECameraScreen extends StatefulWidget {
  const PPECameraScreen({super.key});
  @override
  State<PPECameraScreen> createState() => _PPECameraScreenState();
}

class _PPECameraScreenState extends State<PPECameraScreen> {
  late CameraController _controller;
  late Interpreter _interpreter;
  List<String> _labels = [];
  List<String> _detections = [];
  String _status = "Initializing...";
  String _debugInfo = "";
  bool _isReady = false;
  bool _isProcessing = false;
  final int _inputSize = 320;
  final double _confThreshold = 0.25;
  int _numClasses = 80;
  int _numAnchors = 2100;
  bool _outputTransposed = true;
  int _inputTensorSize = 0;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      _status = "1. Camera...";
      if (mounted) setState(() {});
      _controller = CameraController(cameras[0], ResolutionPreset.medium, enableAudio: false);
      await _controller.initialize();

      _status = "2. Model...";
      if (mounted) setState(() {});
      _labels = (await rootBundle.loadString('assets/labels.txt'))
          .split('\n').where((s) => s.trim().isNotEmpty).toList();

      _interpreter = await Interpreter.fromAsset('assets/best.tflite');
      final inputTensor = _interpreter.getInputTensor(0);
      final inputShape = inputTensor.shape;
      _inputTensorSize = inputShape.reduce((a, b) => a * b);
      final outShape = _interpreter.getOutputTensor(0).shape;

      if (outShape.length == 3 && outShape[1] < outShape[2]) {
        _outputTransposed = true;
        _numClasses = outShape[1] - 5;
        _numAnchors = outShape[2];
      } else {
        _outputTransposed = false;
        _numAnchors = outShape[1];
        _numClasses = outShape[2] - 5;
      }

      _debugInfo = "In: $inputShape\nOut: $outShape\nClasses: $_numClasses";
      _isReady = true;
      await Future.delayed(const Duration(seconds: 1));
      if (mounted) setState(() {});

      await _controller.startImageStream(_processFrame);
      if (mounted) {
        _status = "✅ Active";
        setState(() {});
      }
    } catch (e, st) {
      _status = "❌ Init: $e";
      _debugInfo = st.toString().substring(0, 200);
      if (mounted) setState(() {});
    }
  }

  void _processFrame(CameraImage image) {
    if (!_isReady || _isProcessing) return;
    _isProcessing = true;

    Future(() {
      try {
        final input = _cameraImageToFloat32(image);
        if (input.length != _inputTensorSize) return;
        final output = Float32List((4 + 1 + _numClasses) * _numAnchors);
        _interpreter.run(input, output);
        _parseYOLO(output);
      } catch (e) {
        _status = "❌ Run: $e";
      } finally {
        _isProcessing = false;
        if (mounted) setState(() {});
      }
    });
  }

  Float32List _cameraImageToFloat32(CameraImage image) {
    final result = Float32List(_inputSize * _inputSize * 3);
    final yRow = image.planes[0].bytesPerRow;
    final uvRow = image.planes[1].bytesPerRow;
    final uvPixel = uvRow ~/ (image.width ~/ 2);
    final y = image.planes[0].bytes;
    final u = image.planes[1].bytes;
    final v = image.planes[2].bytes;
    int idx = 0;

    for (int ty = 0; ty < _inputSize; ty++) {
      final sy = (ty * image.height ~/ _inputSize);
      final yOff = sy * yRow;
      final uvOff = (sy ~/ 2) * uvRow;
      for (int tx = 0; tx < _inputSize; tx++) {
        final sx = (tx * image.width ~/ _inputSize);
        final uvIdx = uvOff + (sx ~/ 2) * uvPixel;
        final yVal = y[yOff + sx];
        final uVal = u[uvIdx];
        final vVal = v[uvIdx];
        int r = (yVal + 1.370705 * (vVal - 128)).clamp(0, 255);
        int g = (yVal - 0.698001 * (uVal - 128) - 0.337633 * (vVal - 128)).clamp(0, 255);
        int b = (yVal + 1.732446 * (uVal - 128)).clamp(0, 255);
        result[idx++] = r / 255.0;
        result[idx++] = g / 255.0;
        result[idx++] = b / 255.0;
      }
    }
    return result;
  }

  void _parseYOLO(List<double> output) {
    final newDetections = <String>[];
    newDetections.add("Raw: ${output.take(5).map((v) => v.toStringAsFixed(2)).join(' ')}");
    double maxConf = 0;

    for (int j = 0; j < _numAnchors; j++) {
      final objIdx = _outputTransposed ? 4 * _numAnchors + j : j * (5 + _numClasses) + 4;
      if (objIdx >= output.length) continue;
      final objConf = output[objIdx];
      if (objConf > maxConf) maxConf = objConf;
      if (objConf > _confThreshold) {
        double maxScore = -1;
        int maxC = 0;
        for (int c = 0; c < _numClasses; c++) {
          final cIdx = _outputTransposed ? (5 + c) * _numAnchors + j : j * (5 + _numClasses) + 5 + c;
          if (cIdx >= output.length) continue;
          if (output[cIdx] > maxScore) {
            maxScore = output[cIdx];
            maxC = c;
          }
        }
        final score = objConf * maxScore;
        if (score > _confThreshold && maxC < _labels.length) {
          newDetections.add("${_labels[maxC]}: ${(score * 100).toInt()}%");
        }
      }
    }
    newDetections.add("Max: ${maxConf.toStringAsFixed(3)}");
    _detections = newDetections;
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(body: Center(child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [CircularProgressIndicator(), const SizedBox(height: 16), Text(_status)],
      )));
    }
    return Scaffold(body: Stack(children: [
      CameraPreview(_controller),
      Positioned(top: 40, left: 10, right: 10,
        child: Container(padding: const EdgeInsets.all(6), color: Colors.black54,
          child: Text(_status, style: const TextStyle(color: Colors.yellow, fontSize: 11)))),
      Positioned(top: 90, left: 10, right: 10,
        child: Container(padding: const EdgeInsets.all(6), color: Colors.blue[900],
          child: Text(_debugInfo, style: const TextStyle(color: Colors.white, fontSize: 9, fontFamily: 'monospace')))),
      Positioned(bottom: 20, left: 10, right: 10,
        child: Container(padding: const EdgeInsets.all(8), color: Colors.black87,
          child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start,
            children: _detections.map((t) => Padding(
              padding: const EdgeInsets.only(bottom: 2),
              child: Text(t, style: const TextStyle(color: Colors.greenAccent, fontSize: 13, fontWeight: FontWeight.bold)),
            )).toList()))),
    ]));
  }

  @override
  void dispose() {
    _controller.stopImageStream();
    _controller.dispose();
    _interpreter.close();
    super.dispose();
  }
}