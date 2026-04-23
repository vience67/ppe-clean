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
  String _debugRawOutput = "...";
  bool _isReady = false;
  bool _isProcessing = false;
  final int _inputSize = 320;
  final double _confThreshold = 0.05;

  int _numClasses = 3;
  int _numAnchors = 2100;

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
      
      // Загружаем модель
      _interpreter = await Interpreter.fromAsset('assets/best.tflite');
      
      // 🔧 ВАЖНО: Вызываем allocateTensors ОДИН РАЗ здесь
      _interpreter.allocateTensors();

      final outShape = _interpreter.getOutputTensor(0).shape;
      _numClasses = outShape[1] - 5; 
      _numAnchors = outShape[2];
      
      _status = "In: ${_interpreter.getInputTensor(0).shape}\nOut: $outShape\nClasses: $_numClasses";
      _isReady = true;

      await Future.delayed(const Duration(seconds: 2));
      if (mounted) setState(() {});

      bool streamStarted = false;
      for (int i = 0; i < 3; i++) {
        try {
          await _controller.startImageStream(_processFrame);
          streamStarted = true;
          break;
        } catch (e) { await Future.delayed(const Duration(seconds: 1)); }
      }
      
      if (streamStarted) _status += "\n✅ Active";
      if (mounted) setState(() {});

    } catch (e, st) {
      _status = "❌ Init: $e";
      if (mounted) setState(() {});
    }
  }

  void _processFrame(CameraImage image) {
    if (!_isReady || _isProcessing) return;
    _isProcessing = true;

    Future(() {
      try {
        final input = _cameraImageToFloat32(image);
        
        // Размер выхода: Batch(1) * Features(8) * Anchors(2100)
        final int outputSize = 1 * (4 + 1 + _numClasses) * _numAnchors;
        
        // 🔧 Используем Float32List для выхода
        final output = Float32List(outputSize);
        
        // 🔧 ЗАПУСКАЕМ ИНФЕРЕНС (без allocateTensors!)
        _interpreter.run(input, output);
        
        // Выводим первые 10 чисел
        String raw = "Raw[0-9]: ";
        for(int i=0; i<10 && i<output.length; i++) {
           raw += "${output[i].toStringAsFixed(2)} ";
        }
        _debugRawOutput = raw;

        _parseYOLO(output);
        
      } catch (e, st) {
        _status = "❌ TFLite Crash: $e";
        _debugRawOutput = st.toString().substring(0, 150);
        _controller.stopImageStream();
        _isReady = false;
      } finally {
        _isProcessing = false;
        if (mounted) setState(() {});
      }
    });
  }

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

  void _parseYOLO(List<double> output) {
    List<String> newDetections = [];
    
    String confDebug = "ObjConf[0-4]: ";
    for (int j = 0; j < 5; j++) {
       confDebug += "${output[4 * _numAnchors + j].toStringAsFixed(2)} ";
    }
    newDetections.add(confDebug);

    double maxConf = 0;

    for (int j = 0; j < _numAnchors; j++) {
      final objConf = output[4 * _numAnchors + j];

      if (objConf > maxConf) maxConf = objConf;

      if (objConf > _confThreshold) {
        double maxClassScore = -1.0;
        int maxClassIdx = 0;

        for (int c = 0; c < _numClasses; c++) {
          final classScore = output[(5 + c) * _numAnchors + j];
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            maxClassIdx = c;
          }
        }

        final totalScore = objConf * maxClassScore;
        if (totalScore > _confThreshold) {
           final labelName = (maxClassIdx < _labels.length) ? _labels[maxClassIdx] : "C$maxClassIdx";
           newDetections.add('$labelName: ${(totalScore * 100).toInt()}%');
        }
      }
    }
    
    newDetections.add("Max ObjConf: ${maxConf.toStringAsFixed(3)}");
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
        Positioned(top: 50, left: 10, right: 10,
          child: Container(padding: EdgeInsets.all(6), color: Colors.black54,
            child: Text(_status, style: TextStyle(color: Colors.yellowAccent, fontSize: 12, fontWeight: FontWeight.bold)),
          ),
        ),
        Positioned(top: 120, left: 10, right: 10,
          child: Container(padding: EdgeInsets.all(6), color: Colors.blueAccent.withOpacity(0.7),
            child: Text(_debugRawOutput, style: TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'monospace')),
          ),
        ),
        Positioned(bottom: 20, left: 10, right: 10,
          child: Container(padding: EdgeInsets.all(10), color: Colors.black87,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min,
              children: _detections.map((t) => Padding(
                padding: EdgeInsets.only(bottom: 2),
                child: Text(t, style: TextStyle(color: Colors.greenAccent, fontSize: 14, fontFamily: 'monospace', fontWeight: FontWeight.bold))
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