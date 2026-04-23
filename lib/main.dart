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
  String _status = "Loading...";
  bool _isReady = false, _isProcessing = false;
  final int _inputSize = 320;
  final double _confThreshold = 0.1; // Низкий порог для отладки

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      _controller = CameraController(cameras[0], ResolutionPreset.medium, enableAudio: false);
      await _controller.initialize();

      _labels = (await rootBundle.loadString('assets/labels.txt'))
          .split('\n')
          .where((s) => s.trim().isNotEmpty)
          .toList();
      _status = "Labels: ${_labels.length}";

      _interpreter = await Interpreter.fromAsset('assets/best.tflite');
      _status = "Model loaded";

      _isReady = true;
      await _controller.startImageStream(_processFrame);
      if (mounted) setState(() {});
    } catch (e) {
      _status = "❌ Init: $e";
      if (mounted) setState(() {});
    }
  }

  void _processFrame(CameraImage image) {
    if (!_isReady || _isProcessing) return;
    _isProcessing = true;
    _status = "🔄 Processing...";

    Future(() {
      try {
        final input = _cameraImageToFloat32(image);
        final output = List.filled(1 * 84 * 8400, 0.0);
        _interpreter.run(input, output);
        _parseYOLO(output);
        _status = "✅ OK";
      } catch (e, st) {
        _detections = ["❌ $e", st.toString().split('\n').first];
        _status = "⛔ Error";
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
    _detections = [];
    
    // 🔍 Дебаг: первые 20 значений
    String debug = "Out[0-19]: ";
    for (int i = 0; i < 20 && i < output.length; i++) {
      debug += "${output[i].toStringAsFixed(3)} ";
    }
    _detections.add(debug);
    
    // 🔍 Максимальная уверенность
    double maxConf = 0;
    for (int i = 0; i < 8400; i++) {
      final conf = output[i * 84 + 4];
      if (conf > maxConf) maxConf = conf;
    }
    _detections.add("Max conf: ${maxConf.toStringAsFixed(3)}");
    
    // 🔍 Парсинг детекций
    for (int i = 0; i < 8400; i++) {
      final conf = output[i * 84 + 4];
      if (conf > _confThreshold) {
        var maxCls = 0;
        var maxVal = -1.0;
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
    if (!_controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      body: Stack(children: [
        CameraPreview(_controller),
        // Статус сверху
        Positioned(top: 40, left: 10, right: 10,
          child: Text(_status, style: const TextStyle(color: Colors.yellow, fontSize: 16, fontWeight: FontWeight.bold)),
        ),
        // Детекции снизу
        Positioned(bottom: 20, left: 10, right: 10,
          child: Container(
            padding: const EdgeInsets.all(8),
            color: Colors.black87,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: _detections.map((t) => Text(t, style: const TextStyle(color: Colors.white, fontSize: 14))).toList(),
            ),
          ),
        ),
      ]),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    _interpreter.close();
    super.dispose();
  }
}