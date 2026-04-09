#!/usr/bin/env swift

import Foundation
import AVFoundation
import Vision
import CoreImage
import CoreMedia

// ============================================================
// Skeleton Extraction Tool for BiLSTM Training
// Uses VNDetectHumanBodyPoseRequest (identical to iOS)
// ============================================================

// MARK: - COCO Joint Name Map (identical to iOS SkeletonExtractor)

extension VNHumanBodyPoseObservation.JointName {
    var cocoName: String {
        switch self {
        case .nose: return "nose"
        case .leftEye: return "left_eye"
        case .rightEye: return "right_eye"
        case .leftEar: return "left_ear"
        case .rightEar: return "right_ear"
        case .leftShoulder: return "left_shoulder"
        case .rightShoulder: return "right_shoulder"
        case .leftElbow: return "left_elbow"
        case .rightElbow: return "right_elbow"
        case .leftWrist: return "left_wrist"
        case .rightWrist: return "right_wrist"
        case .leftHip: return "left_hip"
        case .rightHip: return "right_hip"
        case .leftKnee: return "left_knee"
        case .rightKnee: return "right_knee"
        case .leftAnkle: return "left_ankle"
        case .rightAnkle: return "right_ankle"
        default: return rawValue.rawValue
        }
    }
}

// MARK: - Data Structures

struct Keypoint: Codable {
    let joint: String
    let x: Float
    let y: Float
    let confidence: Float
}

struct Frame: Codable {
    let frame_index: Int
    let timestamp: Double
    let keypoints: [Keypoint]
}

struct VideoSkeleton: Codable {
    let video_filename: String
    let fps: Int
    let total_frames: Int
    let duration: Double
    let frames: [Frame]
}

// MARK: - Skeleton Extractor

class SkeletonExtractor {
    // The 17 COCO joints (identical to iOS)
    static let targetJoints: [VNHumanBodyPoseObservation.JointName] = [
        .nose,
        .leftEye, .rightEye,
        .leftEar, .rightEar,
        .leftShoulder, .rightShoulder,
        .leftElbow, .rightElbow,
        .leftWrist, .rightWrist,
        .leftHip, .rightHip,
        .leftKnee, .rightKnee,
        .leftAnkle, .rightAnkle
    ]

    private let poseRequest: VNDetectHumanBodyPoseRequest

    init() {
        self.poseRequest = VNDetectHumanBodyPoseRequest()
    }

    func extractSkeletons(from videoURL: URL, fps: Int = 5) -> VideoSkeleton? {
        let asset = AVAsset(url: videoURL)

        guard let videoTrack = asset.tracks(withMediaType: .video).first else {
            print("  ✗ No video track found")
            return nil
        }

        let duration = CMTimeGetSeconds(asset.duration)
        let videoFPS = videoTrack.nominalFrameRate
        let frameInterval = Int(videoFPS / Float(fps))

        guard let reader = try? AVAssetReader(asset: asset) else {
            print("  ✗ Failed to create asset reader")
            return nil
        }

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        reader.add(output)

        guard reader.startReading() else {
            print("  ✗ Failed to start reading")
            return nil
        }

        var frames: [Frame] = []
        var frameIndex = 0
        var processedFrameIndex = 0

        while let sampleBuffer = output.copyNextSampleBuffer() {
            defer { frameIndex += 1 }

            // Sample every N frames
            guard frameIndex % frameInterval == 0 else { continue }

            guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                continue
            }

            let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

            // Extract skeleton
            if let keypoints = extractKeypoints(from: imageBuffer) {
                let frame = Frame(
                    frame_index: processedFrameIndex,
                    timestamp: timestamp,
                    keypoints: keypoints
                )
                frames.append(frame)
                processedFrameIndex += 1
            }
        }

        reader.cancelReading()

        guard !frames.isEmpty else {
            print("  ✗ No frames with detected skeletons")
            return nil
        }

        return VideoSkeleton(
            video_filename: videoURL.lastPathComponent,
            fps: fps,
            total_frames: frames.count,
            duration: duration,
            frames: frames
        )
    }

    private func extractKeypoints(from pixelBuffer: CVPixelBuffer) -> [Keypoint]? {
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )

        do {
            try handler.perform([poseRequest])
        } catch {
            return nil
        }

        guard let observation = poseRequest.results?.first else {
            return nil
        }

        var keypoints: [Keypoint] = []

        for jointName in Self.targetJoints {
            if let recognizedPoint = try? observation.recognizedPoint(jointName) {
                // Vision uses bottom-left origin, flip Y to top-left
                let flippedY = 1.0 - recognizedPoint.location.y

                keypoints.append(Keypoint(
                    joint: jointName.cocoName,
                    x: Float(recognizedPoint.location.x),
                    y: Float(flippedY),
                    confidence: Float(recognizedPoint.confidence)
                ))
            } else {
                // Missing joint - add zero confidence
                keypoints.append(Keypoint(
                    joint: jointName.cocoName,
                    x: 0.0,
                    y: 0.0,
                    confidence: 0.0
                ))
            }
        }

        return keypoints
    }
}

// MARK: - Main Program

func main() {
    print("🎾 Tennis Skeleton Extraction Tool")
    print("=" * 50)

    let fileManager = FileManager.default

    // Get the directory where this executable is located
    let executablePath = CommandLine.arguments[0]
    let executableURL = URL(fileURLWithPath: executablePath)
    let programDir = executableURL.deletingLastPathComponent().path

    let videosDir = URL(fileURLWithPath: programDir).appendingPathComponent("videos")
    let skeletonsDir = URL(fileURLWithPath: programDir).appendingPathComponent("skeletons")

    // Verify directories exist
    guard fileManager.fileExists(atPath: videosDir.path) else {
        print("✗ Error: videos/ directory not found")
        print("  Please create it and add video files")
        exit(1)
    }

    // Create skeletons directory if needed
    try? fileManager.createDirectory(at: skeletonsDir, withIntermediateDirectories: true)

    // Find all video files
    guard let videoFiles = try? fileManager.contentsOfDirectory(
        at: videosDir,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    ) else {
        print("✗ Error: Failed to read videos/ directory")
        exit(1)
    }

    let videoExtensions = ["mp4", "mov", "m4v", "MP4", "MOV", "M4V"]
    let videos = videoFiles.filter { url in
        videoExtensions.contains(url.pathExtension)
    }

    guard !videos.isEmpty else {
        print("✗ No video files found in videos/")
        print("  Supported formats: MP4, MOV, M4V")
        exit(1)
    }

    print("\nFound \(videos.count) video file(s)")
    print("")

    let extractor = SkeletonExtractor()
    var successCount = 0
    var failCount = 0

    for (index, videoURL) in videos.enumerated() {
        let filename = videoURL.lastPathComponent
        print("[\(index + 1)/\(videos.count)] Processing: \(filename)")

        if let skeleton = extractor.extractSkeletons(from: videoURL, fps: 5) {
            // Save JSON
            let outputFilename = videoURL.deletingPathExtension().lastPathComponent + ".json"
            let outputURL = skeletonsDir.appendingPathComponent(outputFilename)

            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let jsonData = try encoder.encode(skeleton)
                try jsonData.write(to: outputURL)

                let sizeKB = Double(jsonData.count) / 1024.0
                print("  ✓ Saved: \(outputFilename) (\(skeleton.total_frames) frames, \(String(format: "%.1f", sizeKB)) KB)")
                successCount += 1
            } catch {
                print("  ✗ Failed to save JSON: \(error.localizedDescription)")
                failCount += 1
            }
        } else {
            print("  ✗ Failed to extract skeletons")
            failCount += 1
        }

        print("")
    }

    print("=" * 50)
    print("✓ Completed: \(successCount) succeeded, \(failCount) failed")
    print("\nJSON files saved to: skeletons/")
}

// Helper for string repetition
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Run main
main()
