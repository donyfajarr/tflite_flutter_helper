import 'package:image/image.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/src/image/color_space_type.dart';
import 'package:tflite_flutter_helper/src/tensorbuffer/tensorbuffer.dart';

class ImageConversions {
  static Image convertRgbTensorBufferToImage(TensorBuffer buffer) {
    List<int> shape = buffer.getShape();
    ColorSpaceType rgb = ColorSpaceType.RGB;
    rgb.assertShape(shape);

    int h = rgb.getHeight(shape);
    int w = rgb.getWidth(shape);
    Image image = Image(w, h);

    List<int> rgbValues = buffer.getIntList();
    assert(rgbValues.length == w * h * 3);

    for (int i = 0, j = 0, wi = 0, hi = 0; j < rgbValues.length; i++) {
      int r = rgbValues[j++];
      int g = rgbValues[j++];
      int b = rgbValues[j++];
      image.setPixelRgba(wi, hi, r, g, b, 255);
      wi++;
      if (wi % w == 0) {
        wi = 0;
        hi++;
      }
    }

    return image;
  }

  static Image convertGrayscaleTensorBufferToImage(TensorBuffer buffer) {
    TensorBuffer uint8Buffer = buffer.getDataType() == TensorType.uint8
        ? buffer
        : TensorBuffer.createFrom(buffer, TensorType.uint8);

    final shape = uint8Buffer.getShape();
    final grayscale = ColorSpaceType.GRAYSCALE;
    grayscale.assertShape(shape);

    final image = Image.fromBytes(
      grayscale.getWidth(shape),
      grayscale.getHeight(shape),
      uint8Buffer.getBuffer().asUint8List(),
      format: Format.rgba,
    );

    return image;
  }

  static void convertImageToTensorBuffer(Image image, TensorBuffer buffer) {
    int w = image.width;
    int h = image.height;
    List<int>? pixelValues = image.getBytes();
    int flatSize = w * h * 3;
    List<int> shape = [h, w, 3];

    if (pixelValues == null) {
      throw ArgumentError('Image data is null.');
    }

    switch (buffer.getDataType()) {
      case TensorType.uint8:
        buffer.loadList(pixelValues, shape: shape);
        break;

      case TensorType.float32:
        List<double> floatArr = List.generate(
          pixelValues.length,
          (i) => pixelValues[i].toDouble(),
        );
        buffer.loadList(floatArr, shape: shape);
        break;

      default:
        throw StateError(
            "${buffer.getDataType()} is unsupported with TensorBuffer.");
    }
  }
}
