//
//  FaceNetProcessor.swift
//  ObjectDetection
//
//  Created by cha on 5/14/25.
//  Copyright © 2025 Y Media Labs. All rights reserved.
//

import UIKit

/**
 * FaceEmbedding represents the face embedding vector and associated metadata.
 * Unlike object detection which provides bounding boxes, FaceNet generates feature vectors.
 */
struct FaceEmbedding {
    let embedding: [Float]     // The 512-dimensional face embedding vector
    let faceRect: CGRect?      // Optional face rectangle (if provided externally)
    let distance: Float?       // Distance to compare face (used for recognition, optional)
    let personId: String?      // Matched person ID (when comparing against known faces)
}

class FaceNetProcessor: NSObject {
    // FaceNet model configuration
    static let embeddingDimension = 512  // FaceNet outputs a 512-length embedding vector
    
    // Threshold for face matching (cosine similarity threshold)
    static let matchThreshold: Float = 0.7  // -1 ~ 1 사이의 값 중 matchThreshold 값 이상일 경우 일치하는 얼굴로 판정함
    
    /**
     * Process raw FaceNet output into a usable face embedding
     */
    static func outputToFaceEmbedding(outputs: [NSNumber]) -> FaceEmbedding? {
        // Verify we have the correct output size
        guard outputs.count == embeddingDimension else {
            print("Error: Output dimension mismatch. Expected \(embeddingDimension), got \(outputs.count)")
            return nil
        }
        
        // Convert NSNumber array to Float array
        let embedding = outputs.map { Float(truncating: $0) }
        
        // L2 normalize the vector (important for proper face comparison)
        let normalizedEmbedding = normalize(vector: embedding)
        
        // embedding 결과값 로그
//        let formatted = normalizedEmbedding.map { "\($0)f" }
//        print(formatted.joined(separator: ", "))
        return FaceEmbedding(
            embedding: normalizedEmbedding,
            faceRect: nil,
            distance: nil,
            personId: nil
        )
    }
    
    /**
     * L2 normalization of vector (essential for proper face comparison)
     */
    static func normalize(vector: [Float]) -> [Float] {
        // Calculate squared sum
        let squaredSum = vector.reduce(0) { $0 + $1 * $1 }
        
        // Calculate magnitude (square root of sum of squares)
        let magnitude = sqrtf(squaredSum)
        
        // Avoid division by zero
        if magnitude <= 0 {
            return vector
        }
        
        // Normalize each element
        return vector.map { $0 / magnitude }
    }
    
    /**
     * Calculate similarity between two face embeddings (cosine similarity)
     * Returns a value between -1 and 1, where 1 means identical
     */
    static func calculateSimilarity(between embedding1: [Float], and embedding2: [Float]) -> Float {
        guard embedding1.count == embedding2.count else {
            print("Error: Cannot calculate similarity between embeddings of different dimensions")
            return -1.0
        }
        
        // Calculate dot product
        var dotProduct: Float = 0
        for i in 0..<embedding1.count {
            dotProduct += embedding1[i] * embedding2[i]
        }
        
        // With normalized vectors, dot product equals cosine similarity
        return dotProduct
    }
    
    /**
     * Compare a face embedding against a database of known faces
     * Returns the ID and distance of the best match if it exceeds the threshold
     */
    static func findBestMatch(for faceEmbedding: [Float],
                              in knownFaces: [String: [Float]]) -> (personId: String, distance: Float)? {
        var bestMatch: String? = nil
        var bestDistance: Float = -1
        
        for (personId, knownEmbedding) in knownFaces {
            let similarity = calculateSimilarity(between: faceEmbedding, and: knownEmbedding)
            
            if similarity > bestDistance {
                bestDistance = similarity
                bestMatch = personId
            }
        }
        
        // Only return a match if it exceeds our confidence threshold
        if bestDistance >= matchThreshold, let id = bestMatch {
            return (personId: id, distance: bestDistance)
        }
        
        return nil
    }
    
    /**
     * Remove existing face recognition annotations from the view
     */
    static func cleanRecognition(imageView: UIImageView) {
        if let layers = imageView.layer.sublayers {
            for layer in layers {
                if layer is CATextLayer {
                    layer.removeFromSuperlayer()
                }
            }
            for view in imageView.subviews {
                view.removeFromSuperview()
            }
        }
    }
    
    /**
     * Display face recognition results on the image view
     * Note: This requires that face rectangles are provided externally
     * (e.g., from a separate face detection step)
     */
    static func showRecognition(imageView: UIImageView,
                                originalImage: UIImage,
                                faceEmbeddings: [FaceEmbedding]) {
        for face in faceEmbeddings {
            // We can only display if we have a face rectangle
            guard let rect = face.faceRect else { continue }
            
            let imageSize = originalImage.size
            let viewSize = imageView.bounds.size

            let viewRect = convertFaceRect(rect, imageSize: imageSize, imageViewSize: viewSize)
            let bbox = UIView(frame: viewRect)
            
            
            // Create bounding box
//            let bbox = UIView(frame: rect)
            bbox.backgroundColor = UIColor.clear
            bbox.layer.borderColor = UIColor.green.cgColor
            bbox.layer.borderWidth = 2
            imageView.addSubview(bbox)
            
            
            // Add text label if we have an identified person
            if let personId = face.personId, let distance = face.distance {
                let textLayer = CATextLayer()
                textLayer.string = String(format: " %@ %.2f", personId, distance)
                textLayer.foregroundColor = UIColor.white.cgColor
                textLayer.backgroundColor = UIColor.clear.cgColor
                textLayer.fontSize = 14
                textLayer.frame = CGRect(x: rect.origin.x, y: rect.origin.y, width: 150, height: 20)
                imageView.layer.addSublayer(textLayer)
            }
        }
    }
    
    static func convertFaceRect(_ faceRect: CGRect, imageSize: CGSize, imageViewSize: CGSize) -> CGRect {
        // Core Image 기준 좌표 -> UIKit 기준 좌표 (상하 반전)
        let scaleX = imageViewSize.width / imageSize.width
        let scaleY = imageViewSize.height / imageSize.height

        var convertedRect = faceRect
        convertedRect.origin.y = imageSize.height - faceRect.origin.y - faceRect.height
        convertedRect.origin.x *= scaleX
        convertedRect.origin.y *= scaleY
        convertedRect.size.width *= scaleX
        convertedRect.size.height *= scaleY

        return convertedRect
    }
}
