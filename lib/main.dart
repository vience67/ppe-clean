import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite/tflite.dart';
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
  List<String> _labels = [];
  List<String> _detections = [];
  String _status = "Initializing...";
  String _debugRawOutput = "...";
  bool _isReady = false;
  bool _isProcessing = false;
  final int _inputSize = 320;
  final double _confThreshold = 0.1;

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

      // 🔥 ЗАГРУЗКА МОДЕЛИ
      var res = await Tflite.loadModel(
        model: "assets/best.tflite",
        labels: "assets/labels.txt",
        numThreads: 1,
        isAsset: true,
        useGpuDelegate: false,
      );
      
      if (res == null) throw Exception("Failed to load model");

      _status = "Model Loaded\nLabels: ${_labels.length}";
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

    Future(() async {
      try {
        // 🔥 ИСПОЛЬЗУЕМ detectObjectOnFrame
        var recognitions = await Tflite.detectObjectOnFrame(
          image.planes[0].bytes,
          model: "SSD", 
          imageHeight: _inputSize,
          imageWidth: _inputSize,
          imageMean: 127.5,
          imageStd: 127.5,
          rotation: 0,
          numResultsPerClass: 5,
          threshold: 0.1,
        );

        if (recognitions != null && recognitions.isNotEmpty) {
          _detections = [];
          for (var rec in recognitions) {
            _detections.add("${rec['detectedClass']}: ${(rec['confidenceInClass'] * 100).toInt()}%");
          }
          _debugRawOutput = "Found: ${recognitions.length}";
        } else {
          _debugRawOutput = "No detections";
        }

      } catch (e, st) {
        _status = "❌ TFLite: $e";
        _debugRawOutput = st.toString().substring(0, 100);
      } finally {
        _isProcessing = false;
        if (mounted) setState(() {});
      }
    });
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
    Tflite.close();
    super.dispose();
  }
}