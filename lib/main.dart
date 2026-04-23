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
  final double _confThreshold = 0.1;

  // Параметры модели
  bool _inputIsUint8 = false;
  int _numClasses = 3;
  int _numAnchors = 2100;
  bool _outputTransposed = true; // [1, 8, 2100] вместо [1, 2100, 8]

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

      // 🔍 Читаем тип входа
      final inputTensor = _interpreter.getInputTensor(0);
      _inputIsUint8 = inputTensor.type == TfLiteType.uint8;
      
      // 🔍 Читаем форму выхода
      final outShape = _interpreter.getOutputTensor(0).shape;
      // [1, 8, 2100] → transposed
      if (outShape.length == 3 && outShape[1] < outShape[2]) {
        _outputTransposed = true;
        _numClasses = outShape[1] - 5;
        _numAnchors = outShape[2];
      } else {
        _outputTransposed = false;
        _numAnchors = outShape[1];
        _numClasses = outShape[2] - 5;
      }

      _debugInfo = "In: ${inputTensor.shape} (${_inputIsUint8 ? 'uint8' : 'float32'})\n";
      _debugInfo += "Out: $outShape (transposed: $_outputTransposed)\n";
      _debugInfo += "Classes: $_numClasses, Anchors: $_numAnchors";
      
      _status = "Model OK\n$_debugInfo";
      _isReady = true;

      await Future.delayed(const Duration(seconds: 2));
      if (mounted) setState(() {});

      // Запуск потока
      bool started = false;
      for (int i = 0; i < 3; i++) {
        try {
          await _controller.startImageStream(_processFrame);
          started = true;
          break;
        } catch (_) => await Future.delayed(const Duration(seconds: 1));
      }
      if (started && mounted) {
        _status += "\n✅ Stream Active";
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
        // Конвертация в нужный тип
        if (_inputIsUint8) {
          final input = _cameraImageToUint8(image);
          final output = Uint8List(1 * (4 + 1 + _numClasses) * _numAnchors);
          _interpreter.run(input, output);
          _parseYOLO(_convertUint8ToDouble(output));
        } else {
          final input = _cameraImageToFloat32(image);
          final output = Float32List(1 * (4 + 1 + _numClasses) * _numAnchors);
          _interpreter.run(input, output);
          _parseYOLO(output);
        }
      } catch (e, st) {
        _status = "❌ Run: $e";
        _debugInfo = st.toString().substring(0, 150);
      } finally {
        _isProcessing = false;
        if (mounted) setState(() {});
      }
    });
  }

  // Конвертация для uint8 входа
  Uint8List _cameraImageToUint8(CameraImage image) {
    final int target = _inputSize;
    final Uint8List result = Uint8List(target * target * 3);
    
    final int yRow = image.planes[0].bytesPerRow;
    final int uvRow = image.planes[1].bytesPerRow;
    final int uvPixel = uvRow ~/ (image.width ~/ 2);
    final Uint8List y = image.planes[0].bytes;
    final Uint8List u = image.planes[1].bytes;
    final Uint8List v = image.planes[2].bytes;
    
    int idx = 0;
    for (int ty = 0; ty < target; ty++) {
      final int sy = (ty * image.height / target).floor();
      final int yOff = sy * yRow;
      final int uvOff = (sy ~/ 2) * uvRow;
      for (int tx = 0; tx < target; tx++) {
        final int sx = (tx * image.width / target).floor();
        final int uvIdx = uvOff + (sx ~/ 2) * uvPixel;
        final int yVal = y[yOff + sx];
        final int uVal = u[uvIdx];
        final int vVal = v[uvIdx];
        int r = (yVal + 1.370705 * (vVal - 128)).round().clamp(0, 255);
        int g = (yVal - 0.698001 * (uVal - 128) - 0.337633 * (vVal - 128)).round().clamp(0, 255);
        int b = (yVal + 1.732446 * (uVal - 128)).round().clamp(0, 255);
        result[idx++] = r;
        result[idx++] = g;
        result[idx++] = b;
      }
    }
    return result;
  }

  // Конвертация для float32 входа
  Float32List _cameraImageToFloat32(CameraImage image) {
    final int target = _inputSize;
    final Float32List result = Float32List(target * target * 3);
    
    final int yRow = image.planes[0].bytesPerRow;
    final int uvRow = image.planes[1].bytesPerRow;
    final int uvPixel = uvRow ~/ (image.width ~/ 2);
    final Uint8List y = image.planes[0].bytes;
    final Uint8List u = image.planes[1].bytes;
    final Uint8List v = image.planes[2].bytes;
    
    int idx = 0;
    for (int ty = 0; ty < target; ty++) {
      final int sy = (ty * image.height / target).floor();
      final int yOff = sy * yRow;
      final int uvOff = (sy ~/ 2) * uvRow;
      for (int tx = 0; tx < target; tx++) {
        final int sx = (tx * image.width / target).floor();
        final int uvIdx = uvOff + (sx ~/ 2) * uvPixel;
        final int yVal = y[yOff + sx];
        final int uVal = u[uvIdx];
        final int vVal = v[uvIdx];
        int r = (yVal + 1.370705 * (vVal - 128)).round().clamp(0, 255);
        int g = (yVal - 0.698001 * (uVal - 128) - 0.337633 * (vVal - 128)).round().clamp(0, 255);
        int b = (yVal + 1.732446 * (uVal - 128)).round().clamp(0, 255);
        result[idx++] = r / 255.0;
        result[idx++] = g / 255.0;
        result[idx++] = b / 255.0;
      }
    }
    return result;
  }

  List<double> _convertUint8ToDouble(Uint8List input) {
    return input.map((e) => e / 255.0).toList();
  }

  void _parseYOLO(List<double> output) {
    List<String> newDetections = [];
    
    // Дебаг
    String debug = "Raw[0-4]: ";
    for (int i = 0; i < 5 && i < output.length; i++) {
      debug += "${output[i].toStringAsFixed(2)} ";
    }
    newDetections.add(debug);

    double maxConf = 0;

    for (int j = 0; j < _numAnchors; j++) {
      // Читаем уверенность объекта (feature #4)
      final int objIdx = _outputTransposed 
          ? 4 * _numAnchors + j 
          : j * (4 + 1 + _numClasses) + 4;
      
      if (objIdx >= output.length) continue;
      final objConf = output[objIdx];

      if (objConf > maxConf) maxConf = objConf;

      if (objConf > _confThreshold) {
        double maxClassScore = -1.0;
        int maxClassIdx = 0;

        for (int c = 0; c < _numClasses; c++) {
          final int classIdx = _outputTransposed
              ? (5 + c) * _numAnchors + j
              : j * (4 + 1 + _numClasses) + 5 + c;
          
          if (classIdx >= output.length) continue;
          final classScore = output[classIdx];
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            maxClassIdx = c;
          }
        }

        final totalScore = objConf * maxClassScore;
        if (totalScore > _confThreshold) {
          final labelName = (maxClassIdx < _labels.length) 
              ? _labels[maxClassIdx] 
              : "C$maxClassIdx";
          newDetections.add('$labelName: ${(totalScore * 100).toInt()}%');
        }
      }
    }
    
    newDetections.add("MaxConf: ${maxConf.toStringAsFixed(3)}");
    _detections = newDetections;
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller.value.isInitialized) {
      return Scaffold(body: Center(child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        CircularProgressIndicator(), SizedBox(height: 16), Text(_status)
      ])));
    }
    return Scaffold(
      body: Stack(children: [
        CameraPreview(_controller),
        Positioned(top: 40, left: 10, right: 10,
          child: Container(padding: EdgeInsets.all(6), color: Colors.black54,
            child: Text(_status, style: TextStyle(color: Colors.yellow, fontSize: 11)),
          ),
        ),
        Positioned(top: 110, left: 10, right: 10,
          child: Container(padding: EdgeInsets.all(6), color: Colors.blue[900],
            child: Text(_debugInfo, style: TextStyle(color: Colors.white, fontSize: 10, fontFamily: 'monospace')),
          ),
        ),
        Positioned(bottom: 20, left: 10, right: 10,
          child: Container(padding: EdgeInsets.all(10), color: Colors.black87,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min,
              children: _detections.map((t) => Padding(
                padding: EdgeInsets.only(bottom: 2),
                child: Text(t, style: TextStyle(color: Colors.greenAccent, fontSize: 14, fontWeight: FontWeight.bold))
              )).toList(),
            ),
          ),
        ),
      ]),
    );
  }

  @override
  void dispose() {
    _controller.stopImageStream();
    _controller.dispose();
    _interpreter.close();
    super.dispose();
  }
}