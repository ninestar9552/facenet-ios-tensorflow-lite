//
//  FaceNetModelHandler.swift
//  ObjectDetection
//
//  Created by cha on 5/14/25.
//  Copyright © 2025 Y Media Labs. All rights reserved.
//

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct FaceNetResult {
    let inferenceTime: Double
    let faceEmbedding: FaceEmbedding?
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the FaceNet model.
enum FaceNet {
    static let modelInfo: FileInfo = (name: "facenet_512", extension: "tflite")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then processes the FaceNet output embedding.
class FaceNetModelHandler: NSObject {
    
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int
    let threadCountLimit = 10
    
    // MARK: Model parameters
    let batchSize = 4
    let inputChannels = 3
    let inputWidth = 160  // FaceNet typically expects 160x160 images
    let inputHeight = 160
    
    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter
    
    // MARK: - Initialization
    
    /// A failable initializer for `FaceNetModelHandler`. A new instance is created if the model and
    /// Default `threadCount` is 1.
    init?(modelFileInfo: FileInfo, threadCount: Int = 1) {
        let modelFilename = modelFileInfo.name
        
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        // Specify the options for the `Interpreter`.
        self.threadCount = threadCount
        var options = Interpreter.Options()
        options.threadCount = threadCount
        do {
            // Create the `Interpreter`.
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        super.init()
    }
    
    /// This method runs the FaceNet model on a given frame and returns the face embedding
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> FaceNetResult? {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
//        print("runModel! width: \(imageWidth), height: \(imageHeight)")
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = pixelBuffer.resizedForFaceNet(to: scaledSize) else {
            return nil
        }
        
        let interval: TimeInterval
        let outputTensor: Tensor

        do {
            let inputTensor = try interpreter.input(at: 0)
//            print("Input tensor shape: \(inputTensor.shape), type: \(inputTensor.dataType)")
            let expectedByteCount = batchSize * inputWidth * inputHeight * inputChannels
//            print("Expected byte count: \(expectedByteCount)")
            
            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
                byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
//            print("RGB data size: \(rgbData.count)")
            
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            let startDate = Date()
            try interpreter.invoke()
            interval = Date().timeIntervalSince(startDate) * 1000
            
            // Get the output tensor
            outputTensor = try interpreter.output(at: 0)
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Convert output to NSNumber array
        guard let outputs = [Float](unsafeData: outputTensor.data) as [NSNumber]? else {
            print("Failed to convert output tensor data")
            return nil
        }
        
        // Process the raw output into a face embedding
        let faceEmbedding = FaceNetProcessor.outputToFaceEmbedding(outputs: outputs)
        
        let result = FaceNetResult(inferenceTime: interval, faceEmbedding: faceEmbedding)
        return result
    }
    
    
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                       height: vImagePixelCount(height),
                                       width: vImagePixelCount(width),
                                       rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                            height: vImagePixelCount(height),
                                            width: vImagePixelCount(width),
                                            rowBytes: destinationBytesPerRow)
        
        if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        if isModelQuantized {
            return byteData
        }
        
        // Not quantized, convert to floats and standardize
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        
        // First pass: convert bytes to floats
        for i in 0..<bytes.count {
            floats.append(Float(bytes[i]))
        }
        
        // Calculate mean
        var sum: Float = 0.0
        for pixel in floats {
            sum += pixel
        }
        let mean = sum / Float(floats.count)
        
        // Calculate standard deviation
        var sumSquaredDiff: Float = 0.0
        for pixel in floats {
            let diff = pixel - mean
            sumSquaredDiff += diff * diff
        }
        
        var std = sqrt(sumSquaredDiff / Float(floats.count))
        
        // Set minimum standard deviation to prevent division by very small numbers
        let minStd = 1.0 / sqrt(Float(floats.count))
        std = max(std, minStd)
        
        // Standardize each pixel
        for i in 0..<floats.count {
            floats[i] = (floats[i] - mean) / std
        }
        
        return Data(copyingBufferOf: floats)
    }
}

// MARK: - Extensions

extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    /// Creates a new array from the bytes of the given unsafe data.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
#if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
#else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
#endif  // swift(>=5.0)
    }
}

// Extension for resizing pixel buffers
extension CVPixelBuffer {
    /// FaceNet 모델을 위한 정확한 크기 리사이즈 (aspect ratio 무시)
    func resizedForFaceNet(to size: CGSize) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(size.width),
                                         Int(size.height),
                                         kCVPixelFormatType_32BGRA,
                                         attrs,
                                         &pixelBuffer)
        
        guard status == kCVReturnSuccess, let resultBuffer = pixelBuffer else {
            return nil
        }
        
        let ciImage = CIImage(cvPixelBuffer: self)
        let sourceWidth = ciImage.extent.width
        let sourceHeight = ciImage.extent.height
        let targetWidth = size.width
        let targetHeight = size.height
        
        // 🔥 핵심: aspect ratio 무시하고 정확히 타겟 크기로 변환
        let scaleX = targetWidth / sourceWidth
        let scaleY = targetHeight / sourceHeight
        
        let scaledImage = ciImage
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        let context = CIContext()
        CVPixelBufferLockBaseAddress(resultBuffer, .readOnly)
        
        // 정확히 전체 영역을 채우도록 렌더링
        let targetRect = CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
        context.render(scaledImage, to: resultBuffer, bounds: targetRect, colorSpace: CGColorSpaceCreateDeviceRGB())
        
        CVPixelBufferUnlockBaseAddress(resultBuffer, .readOnly)
        
        return resultBuffer
    }
}
