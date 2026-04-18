import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
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
    title: 'PPE Detector', theme: ThemeData.dark(), home: const PPECameraScreen());
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
  bool _isReady = false, _isProcessing = false;
  final int _inputSize = 320;
  final double _confThreshold = 0.1;

  @override
  void initState() { super.initState(); _init(); }

  Future<void> _init() async {
    _controller = CameraController(cameras[0], ResolutionPreset.medium, enableAudio: false);
    await _controller.initialize();
    _labels = (await rootBundle.loadString('assets/labels.txt'))
        .split('\n').where((s) => s.trim().isNotEmpty).toList();
    _interpreter = await Interpreter.fromAsset('assets/best.tflite');
    _isReady = true;
    await _controller.startImageStream(_processFrame);
    if (mounted) setState(() {});
  }

  void _processFrame(CameraImage image) {
    if (!_isReady || _isProcessing) return;
    _isProcessing = true;
    Future.delayed(const Duration(milliseconds: 200), () => _isProcessing = false);
    try {
      final buf = _yuv420ToUint8(image);
      final decoded = img.decodeImage(buf)!;
      final resized = img.copyResize(decoded, width: _inputSize, height: _inputSize, interpolation: img.Interpolation.nearest);
      final input = _imageToFloat32(resized);
      
      // 🔧 ИСПРАВЛЕНИЕ: Интерпретатору нужен вход и выход
      final output = List.filled(1 * 84 * 8400, 0.0);
      _interpreter.run(input, output);
      
      _parseYOLO(output);
      if (mounted) setState(() {});
    } catch (_) {}
  }

  Uint8List _yuv420ToUint8(CameraImage image) {
    final yRow = image.planes[0].bytesPerRow;
    final uvRow = image.planes[1].bytesPerRow;
    final uvPixel = image.planes[1].bytesPerRow ~/ (image.width ~/ 2);
    final buf = Uint8List(image.width * image.height * 3);
    int idx = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final uvIdx = uvPixel * (x ~/ 2) + uvRow * (y ~/ 2);
        buf[idx++] = image.planes[0].bytes[y * yRow + x];
        buf[idx++] = image.planes[1].bytes[uvIdx];
        buf[idx++] = image.planes[2].bytes[uvIdx];
      }
    }
    return buf;
  }

  Float32List _imageToFloat32(img.Image image) {
    final pixels = image.getBytes(order: img.ChannelOrder.rgb);
    final buf = Float32List(1 * _inputSize * _inputSize * 3);
    for (int i = 0; i < pixels.length; i++) buf[i] = pixels[i] / 255.0;
    return buf;
  }

  void _parseYOLO(List<double> output) {
    _detections = [];
    for (int i = 0; i < 8400; i++) {
      final conf = output[i * 84 + 4];
      if (conf > _confThreshold) {
        var maxCls = 0; var maxVal = -1.0;
        for (int c = 0; c < _labels.length; c++) {
          final v = output[i * 84 + 5 + c];
          if (v > maxVal) { maxVal = v; maxCls = c; }
        }
        _detections.add('${_labels[maxCls]}: ${(conf * 100).toInt()}%');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    return Scaffold(
      body: Stack(children: [
        CameraPreview(_controller),
        Positioned(bottom: 20, left: 10, right: 10,
          child: Container(padding: const EdgeInsets.all(8), color: Colors.black54,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start,
              children: _detections.map((t) => Text(t, style: const TextStyle(color: Colors.white, fontSize: 16))).toList()),
          ),
        ),
      ]),
    );
  }

  @override
  void dispose() { _controller.dispose(); _interpreter.close(); super.dispose(); }
}