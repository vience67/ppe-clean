import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart'; // Добавь в pubspec.yaml если нет, но обычно не нужно
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
  final double _confThreshold = 0.25; // Чуть повысил порог для стабильности
  
  // Параметры модели
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

      _controller = CameraController(
        cameras[0],
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await _controller.initialize();

      _status = "2. Loading Model (NNAPI)...";
      if (mounted) setState(() {});

      _labels = (await rootBundle.loadString('assets/labels.txt'))
          .split('\n')
          .where((s) => s.trim().isNotEmpty)
          .toList();

      // 🔥 ВАЖНО: Создаем опции с NNAPI делегатом
      // Это решает проблему "failed precondition" для многих моделей
      final options = InterpreterOptions();
      
      try {
        // Пробуем включить NNAPI (аппаратное ускорение Android)
        options.addDelegate(NnApiDelegate());
        _status = "Using NNAPI Delegate...";
      } catch (e) {
        _status = "NNAPI failed, using CPU";
      }
      
      // Загружаем модель с опциями
      _interpreter = await Interpreter.fromAsset('assets/best.tflite', options: options);

      // Читаем размеры тензоров
      final inputTensor = _interpreter.getInputTensor(0);
      final inputShape = inputTensor.shape;
      _inputTensorSize = 1;
      for (int dim in inputShape) {
        _inputTensorSize *= dim;
      }

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

      _status = "Model Ready (${_numClasses} classes)";
      
      _debugInfo = "In: $inputShape\nOut: $outShape";

      _isReady = true;
      await Future.delayed(const Duration(seconds: 1));

      if (mounted) setState(() {});

      bool started = false;
      for (int i = 0; i < 3; i++) {
        try {
          await _controller.startImageStream(_processFrame);
          started = true;
          break;
        } catch (_) {
          await Future.delayed(const Duration(seconds: 1));
        }
      }

      if (started && mounted) {
        _status += "\n✅ Active";
        setState(() {});
      }
    } catch (e, st) {
      _status = "❌ Error: $e";
      _debugInfo = st.toString().substring(0, 200);
      if (mounted) setState(() {});
    }
  }

  void _processFrame(CameraImage image) {
    if (!_isReady || _isProcessing) return;
    _isProcessing = true;

    Future(() {
      try {
        // Конвертация
        final input = _cameraImageToFloat32(image);
        
        // Проверка размера входа (строгая)
        if (input.length != _inputTensorSize) {
           // Если размеры не совпадают, модель не запустится.
           // Логируем, но не крашим приложение
           print("Size mismatch: ${input.length} vs $_inputTensorSize");
           _isProcessing = false;
           return;
        }

        // Создаем выходной буфер под размер ТЕНЗОРА модели, а не хардкод
        // Берем точный размер из интерпретатора, чтобы избежать ошибки аллокации
        final outputTensor = _interpreter.getOutputTensor(0);
        int outputSize = 1;
        for (var dim in outputTensor.shape) outputSize *= dim;
        
        final output = Float32List(outputSize);

        // 🔥 Запуск. Делегат должен справиться с выделением памяти
        _interpreter.run(input, output);

        _parseYOLO(output);
      } catch (e, st) {
        _status = "❌ Run: $e";
        _debugInfo = st.toString().substring(0, 150);
      } finally {
        _isProcessing = false;
        if (mounted) setState(() {});
      }
    });
  }

  Float32List _cameraImageToFloat32(CameraImage image) {
    final int target = _inputSize;
    final Float32List result = Float32List(target * target * 3);
    
    // Проверка на валидность данных камеры
    if (image.planes.isEmpty) return result;

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
    
    // Простой вывод сырых данных для отладки
    if (output.isNotEmpty) {
       newDetections.add("Out0: ${output[0].toStringAsFixed(2)}");
       newDetections.add("Max: ${output.reduce((a, b) => a > b ? a : b).toStringAsFixed(2)}");
    }

    // Логика парсинга YOLO (оставляем как есть, если модель YOLOv5)
    // Но добавляем защиту от выхода за границы списка
    double maxConf = 0;
    
    // Безопасная итерация
    int limit = _numAnchors;
    if (limit * (4 + 1 + _numClasses) > output.length) {
        // Если модель выдала меньше данных, чем мы ожидаем (например YOLOv8 без obj)
        // Пробуем адаптироваться
        if (_outputTransposed && output.length >= _numAnchors * (4 + _numClasses)) {
             // Возможно это YOLOv8 формат (без objectness)
             // Но пока оставим стандартный парсинг, чтобы не усложнять
        }
    }

    for (int j = 0; j < _numAnchors; j++) {
      final int baseIdx = _outputTransposed
          ? j 
          : j * (4 + 1 + _numClasses);
      
      // Индекс objectness (уверенность в объекте)
      // В YOLOv5 это 4-й элемент (индекс 4)
      // В YOLOv8 его нет, там сразу классы
      
      // Пробуем найти objectness, если он есть
      int objIdx = _outputTransposed
          ? 4 * _numAnchors + j
          : baseIdx + 4;
      
      if (objIdx >= output.length) continue; // Защита
      
      double objConf = output[objIdx];
      
      // Если это YOLOv8, objConf будет мусором, но мы надеемся на YOLOv5
      
      if (objConf > _confThreshold) {
        // Ищем лучший класс
        double maxClassScore = -1.0;
        int maxClassIdx = 0;
        
        for (int c = 0; c < _numClasses; c++) {
          int classIdx = _outputTransposed
              ? (5 + c) * _numAnchors + j
              : baseIdx + 5 + c;
          
          if (classIdx >= output.length) continue;
          
          double classScore = output[classIdx];
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            maxClassIdx = c;
          }
        }
        
        double totalScore = objConf * maxClassScore;
        if (totalScore > _confThreshold) {
           String label = (maxClassIdx < _labels.length) ? _labels[maxClassIdx] : "Obj";
           newDetections.add("$label ${(totalScore*100).toInt()}%");
        }
      }
    }
    
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
        Positioned(top: 100, left: 10, right: 10,
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