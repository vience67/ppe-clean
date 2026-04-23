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
  bool _isReady = false;
  bool _isProcessing = false;
  bool _streamStarted = false; // 🔑 Флаг, чтобы не запустить дважды
  final int _inputSize = 320;
  final double _confThreshold = 0.1;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _status = "Camera init...";
      if (mounted) setState(() {});

      // 🔑 Используем MEDIUM (низкое разрешение часто вызывает краш)
      _controller = CameraController(cameras[0], ResolutionPreset.medium, enableAudio: false);
      await _controller.initialize();

      _status = "Loading model...";
      if (mounted) setState(() {});

      _labels = (await rootBundle.loadString('assets/labels.txt'))
          .split('\n').where((s) => s.trim().isNotEmpty).toList();
      
      _interpreter = await Interpreter.fromAsset('assets/best.tflite');
      _isReady = true;

      // 🔑 Запускаем поток ТОЛЬКО после того, как виджет отрисовался
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted && !_streamStarted && _controller.value.isInitialized) {
          _startStream();
        }
      });
      
    } catch (e, st) {
      _status = "❌ Error: $e";
      if (mounted) setState(() {});
    }
  }

  void _startStream() {
    try {
      _streamStarted = true;
      _status = "Stream starting...";
      
      _controller.startImageStream((CameraImage image) {
        if (!_isReady || _isProcessing) return;
        _isProcessing = true;
        _status = "🔄 Detecting...";

        // Асинхронная обработка, чтобы не блокировать поток камеры
        Future(() {
          try {
            final input = _cameraImageToFloat32(image);
            final output = List.filled(1 * 84 * 8400, 0.0);
            _interpreter.run(input, output);
            _parseYOLO(output);
            _status = "✅ OK";
          } catch (e, st) {
            _detections = ["⚠️ $e"];
            _status = "⛔ Error";
          } finally {
            _isProcessing = false;
            if (mounted) setState(() {});
          }
        });
      });
      
      _status = "📷 Camera Active";
      if (mounted) setState(() {});
      
    } catch (e) {
      _status = "❌ Stream failed: $e";
      if (mounted) setState(() {});
    }
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
    _detections = [];
    
    String debug = "Raw[0-5]: ";
    for (int i = 0; i < 5 && i < output.length; i++) {
      debug += "${output[i].toStringAsFixed(2)} ";
    }
    _detections.add(debug);
    
    double maxConf = 0;
    for (int i = 0; i < 8400; i++) {
      final conf = output[i * 84 + 4];
      if (conf > maxConf) maxConf = conf;
    }
    _detections.add("MaxConf: ${maxConf.toStringAsFixed(3)}");
    
    for (int i = 0; i < 8400; i++) {
      final conf = output[i * 84 + 4];
      if (conf > _confThreshold) {
        int maxCls = 0;
        double maxVal = -1.0;
        for (int c = 0; c < _labels.length; c++) {
          final v = output[i * 84 + 5 + c];
          if (v > maxVal) { maxVal = v; maxCls = c; }
        }
        if (maxCls < _labels.length) {
          _detections.add('${_labels[maxCls]}: ${(conf * 100).toInt()}%');
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // Если камера не готова, показываем загрузку
    if (_controller == null || !_controller.value.isInitialized) {
      return Scaffold(body: Center(child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [CircularProgressIndicator(), SizedBox(height: 20), Text(_status)]
      )));
    }
    
    return Scaffold(
      body: Stack(children: [
        CameraPreview(_controller),
        Positioned(top: 50, left: 10, right: 10,
          child: Container(padding: EdgeInsets.all(6), color: Colors.black54,
            child: Text(_status, style: TextStyle(color: Colors.yellowAccent, fontSize: 14, fontWeight: FontWeight.bold)),
          ),
        ),
        Positioned(bottom: 20, left: 10, right: 10,
          child: Container(padding: EdgeInsets.all(10), color: Colors.black87,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min,
              children: _detections.map((t) => Padding(
                padding: EdgeInsets.only(bottom: 2),
                child: Text(t, style: TextStyle(color: Colors.white, fontSize: 13, fontFamily: 'monospace'))
              )).toList(),
            ),
          ),
        ),
      ]),
    );
  }

  @override
  void dispose() {
    _streamStarted = false;
    _controller.dispose();
    _interpreter.close();
    super.dispose();
  }
}