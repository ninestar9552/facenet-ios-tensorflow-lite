// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit
import Vision

class FaceRecognitionViewController: UIViewController {
    
    // MARK: Storyboards Connections
    @IBOutlet weak var previewView: PreviewView!
    @IBOutlet weak var overlayView: OverlayView!
    
    // MARK: Controllers that manage functionality
    private lazy var cameraFeedManager = CameraFeedManager(previewView: previewView)
    //==========================================================================================
    @IBOutlet weak var imageView: UIImageView!
    
    // FaceNet model handler
    private var faceNetModelHandler: FaceNetModelHandler?
    
    // Property to store the current face being processed
    private var currentFacePixelBuffer: CVPixelBuffer?
    private var currentFaceEmbedding: [Float] = []
    
    // Database of known faces
    private var knownFaces: [String: [Float]] = [:]
    
    private let facenetQueue = DispatchQueue(label: "facenetQueue")
    
    // MARK: View Handling Methods
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Initialize FaceNet model
        guard let modelHandler = FaceNetModelHandler(
            modelFileInfo: FaceNet.modelInfo,
            threadCount: 2
        ) else {
            fatalError("Failed to initialize FaceNet model")
        }
        
        self.faceNetModelHandler = modelHandler
        
        // Load any known faces from storage
        loadKnownFaces()
        
        cameraFeedManager.delegate = self
        overlayView.clearsContextBeforeDrawing = true
        
    }
    @IBAction func touchupupup(_ sender: Any) {
        
//        addCurrentFace()
        print("현재얼굴\n\(currentFaceEmbedding)")
    }
    
    // MARK: - Face Database Management
    private func loadKnownFaces() {
        // load known face embeddings from persistent storage
//        knownFaces = [:]
//        knownFaces = ["순혁":[-0.109191895, 0.7753906, -0.578125, 2.4980469, 1.2597656, 1.1494141, -0.86572266, 1.0595703, -1.3632812, 0.37280273, 0.17883301, 0.7011719, -0.9140625, -0.21704102, -0.14526367, -0.8457031, 0.3671875, 1.1494141, 0.054473877, -2.2246094, -1.3603516, 0.9291992, 1.0136719, 0.3544922, -0.35473633, 1.5078125, 1.4433594, -0.044281006, 1.1220703, 1.0849609, -1.6806641, -1.2568359, -0.31518555, -1.1240234, -0.6796875, 0.8173828, -0.4567871, -0.064941406, -1.3740234, 0.64746094, -1.15625, 1.0332031, 0.65527344, -0.22338867, 0.7011719, 0.28051758, 0.072143555, 1.3681641, -1.0507812, 0.20092773, 0.011756897, 0.70410156, 1.0175781, -0.34521484, -1.2001953, 1.0341797, 0.03277588, 1.0537109, 0.015792847, -0.5541992, 0.66748047, -0.105041504, 0.30395508, -1.0751953, -0.19702148, 0.4169922, -1.1777344, 0.51660156, 0.24951172, 0.95703125, 0.55615234, 2.3105469, -0.0063934326, -0.5571289, 0.54541016, -0.7211914, -1.4433594, -0.50634766, -0.023651123, 0.5751953, 1.1572266, 0.055419922, 0.7128906, 0.69433594, -0.32226562, 1.0683594, -0.47875977, 0.5332031, 0.86816406, -0.27416992, 0.5703125, 0.037902832, 0.13366699, -0.8022461, 1.5361328, -0.10345459, 0.22131348, 0.41357422, -0.59375, 0.22106934, 0.68359375, -1.3261719, -0.3720703, 1.0244141, -0.25463867, 0.34423828, -0.10461426, -0.84375, 0.40576172, -1.2880859, 0.8828125, 1.0380859, -0.6875, -0.58203125, 0.56347656, -1.1748047, 0.3955078, -0.15600586, -0.79785156, 0.013442993, -0.12695312, 0.31591797, -0.34179688, -1.6640625, -0.5332031, -1.9033203, -0.40893555, 1.25, 0.66308594, 0.22521973, -0.11090088, -0.71435547, 1.8125, -0.0637207, -2.0332031, 0.015991211, -0.50146484, 0.52783203, 0.66259766, 3.0097656, 0.10241699, 1.3779297, 0.33032227, 0.5180664, 0.12866211, 0.025131226, 0.5600586, -0.5180664, 0.06060791, 0.5996094, 0.74121094, -0.23205566, -1.1767578, 0.9458008, 0.65185547, -0.24682617, -0.42871094, 0.2409668, 0.20263672, -1.09375, 1.5585938, 0.31762695, -0.87158203, 1.3662109, 0.11590576, 0.53027344, -0.4572754, -0.9189453, -1.4199219, 0.9667969, -0.3798828, -0.5214844, -0.28857422, 0.4350586, 0.74072266, 1.2529297, 0.8833008, -0.26367188, -0.63427734, 0.045562744, -1.0556641, -1.6494141, -0.6689453, 0.6455078, 0.29589844, -0.2590332, 0.61816406, 0.7026367, 0.99072266, 1.0400391, 0.40893555, 0.72558594, 1.3623047, -1.28125, 0.5107422, -0.33789062, -0.5551758, -0.36547852, -0.5786133, 0.7421875, -0.33666992, -0.034423828, -0.7519531, -1.0830078, 0.5083008, 0.9223633, 0.6748047, -0.80078125, -0.3359375, -0.9868164, -0.61328125, -1.171875, 0.81103516, 0.20788574, 0.0014953613, 0.16430664, -1.5693359, 1.3457031, -1.2753906, 0.8847656, -1.8535156, -1.9863281, -0.72558594, -0.46777344, 0.34179688, 0.057556152, -1.0244141, 0.16113281, 0.2133789, -0.21728516, 0.83935547, 1.6455078, -0.62890625, -1.8115234, -0.421875, -1.0166016, 0.6928711, 0.6791992, -0.29760742, -0.040130615, -1.0419922, -0.9736328, -0.3137207, 0.48876953, 1.0166016, 1.4228516, -0.12902832, 0.5, 0.30786133, 1.0380859, 0.21862793, 0.75390625, 0.055511475, 1.0576172, -0.18078613, 0.4921875, -0.7709961, 1.8037109, 1.2578125, 0.5415039, 0.5678711, -0.24401855, 0.7919922, -0.027557373, -0.6953125, -0.7314453, 0.4411621, -0.25073242, -0.6894531, 0.24316406, 0.18701172, -0.11743164, 0.828125, 0.19213867, 0.11682129, -0.7324219, 0.22241211, 0.93310547, 0.034454346, -0.18103027, 0.98095703, -0.079956055, -1.4228516, -1.8300781, -1.8134766, 0.14685059, -0.004180908, 0.9482422, -0.13244629, -1.2099609, -0.036956787, -2.6445312, -0.27001953, 0.71728516, -0.6098633, -1.6953125, -0.7133789, 0.6738281, 0.56933594, -0.27856445, 1.3037109, -0.07678223, 1.4589844, -0.17651367, 1.2148438, -0.99853516, 0.3894043, 0.26538086, -0.75146484, 1.4111328, 0.8540039, 0.9375, 0.06750488, 1.6357422, 2.7851562, 0.98876953, -0.22143555, 0.4729004, -0.94873047, 0.3713379, 1.9345703, 0.26538086, -0.35107422, 1.234375, -0.5722656, 0.6928711, 0.39746094, 0.5727539, 1.4091797, 0.44433594, 1.2617188, -0.42773438, -0.53222656, 1.3505859, -0.3161621, 0.06665039, -0.6220703, 0.88964844, -0.13574219, 0.124694824, 0.77978516, 1.2128906, 1.2119141, 0.6123047, -0.18334961, -0.2376709, -1.1572266, 0.94873047, 1.9960938, -1.0566406, -0.75439453, 0.0390625, -0.81933594, -1.1669922, -1.6201172, 0.4255371, 0.9921875, 0.43701172, 0.20690918, -0.9790039, -0.7416992, 0.072021484, -0.92871094, 0.31201172, -0.3256836, -0.14050293, 1.4365234, -0.4152832, 0.11206055, -0.14050293, -0.9355469, -0.49023438, 0.76464844, -0.10321045, 0.8696289, 1.6171875, 0.056396484, 0.17126465, -1.1005859, -0.9526367, -0.21655273, -0.8261719, -1.0761719, 0.81396484, 1.0683594, -0.11791992, -0.37768555, -1.0791016, 0.09686279, -0.7006836, 0.24621582, 0.87060547, -0.07696533, -1.1484375, -0.74658203, 0.23620605, 1.40625, 0.9248047, -0.093322754, 1.0527344, -1.6650391, 0.9863281, -0.51123047, 1.3037109, 0.13476562, 0.32617188, -0.8618164, -0.4675293, 0.7470703, 1.28125, 0.45166016, 0.58740234, -0.29223633, 0.8535156, 1.0273438, -0.4345703, 0.16052246, 1.1875, -0.5449219, -0.43725586, 0.85058594, 0.1496582, 0.6376953, -0.8466797, -1.1103516, 0.58203125, 1.1767578, 0.25927734, 0.79296875, -0.2578125, -0.15258789, 1.0439453, -1.0576172, -0.34326172, -1.7919922, -0.115722656, -1.0927734, -0.23840332, 0.5229492, 0.42871094, -0.91503906, 0.88378906, -1.4941406, -0.07977295, -1.0878906, 0.72558594, -2.1308594, 1.4638672, -0.8564453, -2.5683594, 0.6767578, 1.0185547, 1.0009766, -1.0712891, 0.8666992, 0.004486084, 0.43798828, 1.5546875, 1.3417969, -0.19921875, 0.3330078, 1.0927734, 0.008636475, 1.2880859, 1.4501953, -0.7211914, 0.28759766, -0.3552246, 0.0770874, -0.96777344, -0.88134766, -1.1542969, 1.4072266, -1.6728516, -1.8818359, 0.21252441, 2.15625, -0.3334961, 0.023406982, 0.28857422, 0.77246094, 1.4619141, 0.21508789, -0.21411133, 0.10101318, 1.1738281, -2.4023438, 0.95654297, -0.6201172, -0.36157227, 0.43139648, -0.08703613, -1.2021484, -1.3076172, 1.7197266, -1.1044922, -0.15161133, 0.8149414, 0.3449707, -0.08996582,  -0.5, 0.7480469, 1.0556641, -0.5595703, -0.64404297, 1.1308594, 0.5073242, 0.89697266, 0.39086914, -1.1025391, 0.6928711, -0.9423828]]
    }
    
    @objc func addCurrentFace() {
        print("얼굴추가!!!")
        // This would be triggered by a button to add the current face to known faces
        guard let pixelBuffer = self.currentFacePixelBuffer,
              let result = faceNetModelHandler?.runModel(onFrame: pixelBuffer),
              let embedding = result.faceEmbedding else {
            print("No valid face to add")
            return
        }
        
        // Show dialog to enter name
        let alert = UIAlertController(title: "Add New Face", message: "Enter name for this face", preferredStyle: .alert)
        alert.addTextField { textField in
            textField.placeholder = "Name"
        }
        
        alert.addAction(UIAlertAction(title: "Save", style: .default) { [weak self] _ in
            guard let name = alert.textFields?.first?.text, !name.isEmpty else { return }
            
            // Add to known faces
            self?.knownFaces[name] = embedding.embedding
        })
        
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        present(alert, animated: true)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        cameraFeedManager.checkCameraConfigurationAndStartSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        cameraFeedManager.stopSession()
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    func presentUnableToResumeSessionAlert() {
        let alert = UIAlertController(
            title: "Unable to Resume Session",
            message: "There was an error while attempting to resume session.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        
        self.present(alert, animated: true)
    }
    
    func detectFaces(in image: CIImage, completion: @escaping ([VNFaceObservation]) -> Void) {
        let request = VNDetectFaceRectanglesRequest { request, error in
            guard let observations = request.results as? [VNFaceObservation], error == nil else {
                completion([])
                return
            }
            completion(observations)
        }
        
        // 카메라 orientation 설정 - 중요!
        request.usesCPUOnly = false
        
        let handler = VNImageRequestHandler(ciImage: image, orientation: .up, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print("Failed to perform face detection: \(error)")
                completion([])
            }
        }
    }

    func convertBoundingBox(_ boundingBox: CGRect, imageSize: CGSize, imageView: UIImageView) -> CGRect {
        let width = boundingBox.width * imageSize.width
        let height = boundingBox.height * imageSize.height
        let x = boundingBox.minX * imageSize.width
        let y = (1 - boundingBox.minY - boundingBox.height) * imageSize.height // Y축 반전
        
        // 만약 imageView의 크기와 이미지 자체의 크기가 다르면 스케일링 필요
        let scaleX = imageView.frame.width / imageSize.width
        let scaleY = imageView.frame.height / imageSize.height

        return CGRect(x: x * scaleX, y: y * scaleY, width: width * scaleX, height: height * scaleY)
    }

    // Vision의 normalized coordinates를 실제 이미지 좌표로 변환 (orientation 고려)
    func convertVisionToImageCoordinates(_ boundingBox: CGRect, pixelBuffer: CVPixelBuffer) -> CGRect {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        // Vision 좌표계: (0,0)이 좌측 하단, (1,1)이 우측 상단
        // 픽셀 버퍼 좌표계: (0,0)이 좌측 상단
        let x = boundingBox.origin.x * CGFloat(imageWidth)
        let y = (1.0 - boundingBox.origin.y - boundingBox.size.height) * CGFloat(imageHeight)
        let width = boundingBox.size.width * CGFloat(imageWidth)
        let height = boundingBox.size.height * CGFloat(imageHeight)
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

// MARK: CameraFeedManagerDelegate Methods
extension FaceRecognitionViewController: CameraFeedManagerDelegate {
    
    func didOutput(pixelBuffer: CVPixelBuffer) {
        // Create CIImage for face detection
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        // Detect faces
        
        self.detectFaces(in: ciImage) { observations in
            guard !observations.isEmpty else {
                // No faces detected
//                print("face를 찾지 못함")
                DispatchQueue.main.async {
//                    self.imageView.image = UIImage(ciImage: ciImage)
                    FaceNetProcessor.cleanRecognition(imageView: self.imageView)
                }
                return
            }
        
//            print("face찾음 \(observations.count)")
            
            // Convert to UIImage for display
            let uiImage = UIImage(ciImage: ciImage)
            let imageSize = uiImage.size
            // Process each detected face
            var recognizedFaces: [FaceEmbedding] = []
            
                
                for observation in observations {
                    // Extract face region
                    let faceRect = observation.boundingBox
//                    print("얼굴 좌표 변경전 (Vision normalized): \(faceRect)")
                    
                    // Vision coordinates를 픽셀 버퍼 좌표로 변환
                    let convertedRect = self.convertVisionToImageCoordinates(faceRect, pixelBuffer: pixelBuffer)
//                    print("얼굴 좌표 변경후 (Pixel coordinates): \(convertedRect)")
                    
                    DispatchQueue.main.async {
                        FaceNetProcessor.cleanRecognition(imageView: self.imageView)
                        let convertedBoundingRect = self.convertBoundingBox(faceRect, imageSize: imageSize, imageView: self.imageView)
                        
                        let bbox = UIView(frame: convertedBoundingRect)
                        bbox.backgroundColor = UIColor.clear
                        bbox.layer.borderColor = UIColor.green.cgColor
                        bbox.layer.borderWidth = 2
                        self.imageView.addSubview(bbox)
                    }
                    
                    
                    // Get cropped face image
                    self.facenetQueue.async {
                        
                        guard let faceCrop = self.cropAndOrientFace(from: pixelBuffer, faceRect: convertedRect) else {
                            print("얼굴 크롭 실패")
                            return
                        }
                        self.currentFacePixelBuffer = faceCrop // 현재 얼굴 저장용
                        
                        
                        // 크롭된 얼굴을 이미지뷰에 표시
//                        DispatchQueue.main.async {
//                            let faceImage = UIImage(ciImage: CIImage(cvPixelBuffer: faceCrop.resizedForFaceNet(to: CGSize(width: 160, height: 160))!))
//                            self.imageView.image = faceImage
//                        }
                    
                        // Get face embedding
                        guard let result = self.faceNetModelHandler?.runModel(onFrame: faceCrop) else { return }
                        //            print("얼굴인식 결과 \(result)")
                        
                        // If we have a valid embedding
                        if var faceEmbedding = result.faceEmbedding {
                            self.currentFaceEmbedding = faceEmbedding.embedding
                            // Try to match with known faces
                            if let match = FaceNetProcessor.findBestMatch(
                                for: faceEmbedding.embedding,
                                in: self.knownFaces
                            ) {
                                // Update with match info
                                faceEmbedding = FaceEmbedding(
                                    embedding: faceEmbedding.embedding,
                                    faceRect: faceRect,
                                    distance: match.distance,
                                    personId: match.personId
                                )
                            } else {
                                // No match found
                                faceEmbedding = FaceEmbedding(
                                    embedding: faceEmbedding.embedding,
                                    faceRect: faceRect,
                                    distance: nil,
                                    personId: "Unknown"
                                )
                            }
                            
                            recognizedFaces.append(faceEmbedding)
                        }
                    }
                }
            //        print("recognizedFaces 갯수: \(recognizedFaces.count)")
            
            // Update UI on main thread
//            DispatchQueue.main.async {
//                //            self.imageView.image = uiImage
//                FaceNetProcessor.cleanRecognition(imageView: self.imageView)
//                FaceNetProcessor.showRecognition(imageView: self.imageView, originalImage: uiImage, faceEmbeddings: recognizedFaces)
//            }
            
        }
    }
    
    // MARK: - Face Operations
    private func cropAndOrientFace(from pixelBuffer: CVPixelBuffer, faceRect: CGRect) -> CVPixelBuffer? {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
//        print("원본 이미지 크기: \(imageWidth) x \(imageHeight)")
//        print("크롭할 얼굴 영역: \(faceRect)")
        
        // 얼굴 영역을 이미지 경계 내로 안전하게 조정
        let safeX = max(0, min(faceRect.origin.x, CGFloat(imageWidth) - 1))
        let safeY = max(0, min(faceRect.origin.y, CGFloat(imageHeight) - 1))
        let safeWidth = min(faceRect.size.width, CGFloat(imageWidth) - safeX)
        let safeHeight = min(faceRect.size.height, CGFloat(imageHeight) - safeY)
        
        let safeRect = CGRect(x: safeX, y: safeY, width: safeWidth, height: safeHeight)
        
        // 최소 크기 확인
        guard safeRect.width > 20 && safeRect.height > 20 else {
            print("얼굴 영역이 너무 작음: \(safeRect)")
            return nil
        }
        
//        print("안전한 크롭 영역: \(safeRect)")
        
        // CIImage 생성
        let originalCIImage = CIImage(cvPixelBuffer: pixelBuffer)
//        print("원본 CIImage extent: \(originalCIImage.extent)")
        
        // 크롭 영역 검증 - CIImage의 extent와 교집합 확인
        let cropRect = safeRect.intersection(originalCIImage.extent)
        guard !cropRect.isEmpty else {
            print("크롭 영역이 이미지와 교집합이 없음")
            return nil
        }
        
//        print("검증된 크롭 영역: \(cropRect)")
        
        // CIImage 크롭
        let croppedCIImage = originalCIImage.cropped(to: cropRect)
//        print("크롭된 CIImage extent: \(croppedCIImage.extent)")
        
        // extent를 원점으로 이동 (중요!)
        let translatedImage = croppedCIImage.transformed(by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))
//        print("변환된 CIImage extent: \(translatedImage.extent)")
        
        // 새로운 픽셀 버퍼 생성
        let cropWidth = Int(cropRect.width)
        let cropHeight = Int(cropRect.height)
        
        var croppedPixelBuffer: CVPixelBuffer?
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer) // 원본과 같은 포맷 사용
        
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            cropWidth,
            cropHeight,
            pixelFormat,
            attributes as CFDictionary,
            &croppedPixelBuffer
        )
        
        guard status == kCVReturnSuccess, let resultBuffer = croppedPixelBuffer else {
            print("픽셀 버퍼 생성 실패: \(status)")
            return nil
        }
        
        // CIContext로 렌더링 - 간단한 옵션 사용
        let context = CIContext()
        
        // 픽셀 버퍼의 extent (0,0에서 시작)
        let destinationRect = CGRect(x: 0, y: 0, width: cropWidth, height: cropHeight)
//        print("대상 렌더링 영역: \(destinationRect)")
        
        context.render(translatedImage, to: resultBuffer, bounds: destinationRect, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        
//        print("얼굴 크롭 성공: \(cropWidth) x \(cropHeight)")
        return resultBuffer
    }
    
    
    
    
    
    private func cropFace(from pixelBuffer: CVPixelBuffer, faceRect: CGRect) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Adjust crop rect to ensure we have enough margin around the face
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        // Make square crop with some padding
        let faceCenterX = faceRect.midX
        let faceCenterY = faceRect.midY
        let faceSize = max(faceRect.width, faceRect.height) * 1.5 // 50% padding
        
        let cropX = max(0, faceCenterX - faceSize/2)
        let cropY = max(0, faceCenterY - faceSize/2)
        let cropWidth = min(CGFloat(imageWidth) - cropX, faceSize)
        let cropHeight = min(CGFloat(imageHeight) - cropY, faceSize)
        
        let cropRect = CGRect(x: cropX, y: cropY, width: cropWidth, height: cropHeight)
        
        // Crop the image
        let croppedCIImage = ciImage.cropped(to: cropRect)
        
        // Convert CIImage to CVPixelBuffer
        var croppedPixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                    kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(cropWidth),
            Int(cropHeight),
            CVPixelBufferGetPixelFormatType(pixelBuffer),
            attrs,
            &croppedPixelBuffer
        )
        
        guard status == kCVReturnSuccess, let resultBuffer = croppedPixelBuffer else {
            return nil
        }
        
        let context = CIContext()
        context.render(croppedCIImage, to: resultBuffer)
        
        return resultBuffer
    }
    
    func presentVideoConfigurationErrorAlert() {
        
        let alertController = UIAlertController(title: "Configuration Failed", message: "Configuration of camera has failed.", preferredStyle: .alert)
        let okAction = UIAlertAction(title: "OK", style: .cancel, handler: nil)
        alertController.addAction(okAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    func presentCameraPermissionsDeniedAlert() {
        
        let alertController = UIAlertController(title: "Camera Permissions Denied", message: "Camera permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
            
            UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
        }
        
        alertController.addAction(cancelAction)
        alertController.addAction(settingsAction)
        
        present(alertController, animated: true, completion: nil)
        
    }
    
    func sessionRunTimeErrorOccurred() {
        
    }
    
    func sessionWasInterrupted(canResumeManually resumeManually: Bool) {
        
    }
    
    func sessionInterruptionEnded() {
        
    }
}
