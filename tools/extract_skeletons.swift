#!/usr/bin/env swift

import Foundation
import AVFoundation
import Vision
import CoreImage
import CoreMedia

// ============================================================
// Skeleton Extraction Tool (30fps) for CTR-GCN Training
// Uses VNDetectHumanBodyPoseRequest (identical to iOS runtime)
//
// Coordinate convention:
//   Vision raw output: bottom-left origin [0,1]
//   Stored in JSON:    top-left origin [0,1]  (Y already flipped)
//   Model input:       additionally requires x↔y swap (done in build_windows.py)
//
// Usage:
//   ./extract_skeletons --videos <dir> --output <dir> [--fps 30]
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

    func extractSkeletons(from videoURL: URL, fps: Int = 30) -> VideoSkeleton? {
        let asset = AVAsset(url: videoURL)

        guard let videoTrack = asset.tracks(withMediaType: .video).first else {
            print("  ✗ No video track found")
            return nil
        }

        let duration = CMTimeGetSeconds(asset.duration)
        let videoFPS = videoTrack.nominalFrameRate
        // When video fps < target fps, keep every frame (interval=1)
        let frameInterval = max(1, Int(round(videoFPS / Float(fps))))

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

func printUsage() {
    print("Usage: extract_skeletons --videos <dir> --output <dir> [--fps 30]")
    print("")
    print("  --videos   Path to directory containing video files (.mov, .mp4, .m4v)")
    print("  --output   Path to output directory for skeleton JSON files")
    print("  --fps      Target extraction frame rate (default: 30)")
    print("")
    print("Example:")
    print("  ./extract_skeletons --videos data/raw_videos --output data/skeletons_30fps")
}

func parseArgs() -> (videosDir: URL, outputDir: URL, fps: Int)? {
    let args = CommandLine.arguments
    var videosPath: String? = nil
    var outputPath: String? = nil
    var fps: Int = 30

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--videos":
            if i + 1 < args.count { videosPath = args[i + 1]; i += 2 } else { return nil }
        case "--output":
            if i + 1 < args.count { outputPath = args[i + 1]; i += 2 } else { return nil }
        case "--fps":
            if i + 1 < args.count, let v = Int(args[i + 1]) { fps = v; i += 2 } else { return nil }
        case "--help", "-h":
            printUsage(); exit(0)
        default:
            i += 1
        }
    }

    // Fallback: if no args, try legacy sibling-directory layout (videos/ & skeletons/)
    if videosPath == nil || outputPath == nil {
        let execDir = URL(fileURLWithPath: CommandLine.arguments[0])
            .deletingLastPathComponent().path
        videosPath  = videosPath  ?? "\(execDir)/videos"
        outputPath  = outputPath  ?? "\(execDir)/skeletons"
    }

    return (
        URL(fileURLWithPath: videosPath!),
        URL(fileURLWithPath: outputPath!),
        fps
    )
}

func main() {
    print("🎾 Tennis Skeleton Extraction Tool (30fps build)")
    print("=" * 50)

    guard let (videosDir, skeletonsDir, targetFPS) = parseArgs() else {
        printUsage()
        exit(1)
    }

    let fileManager = FileManager.default

    guard fileManager.fileExists(atPath: videosDir.path) else {
        print("✗ Error: videos directory not found: \(videosDir.path)")
        exit(1)
    }

    // Create output directory if needed
    try? fileManager.createDirectory(at: skeletonsDir, withIntermediateDirectories: true)

    // Find all video files
    guard let videoFiles = try? fileManager.contentsOfDirectory(
        at: videosDir,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    ) else {
        print("✗ Error: Failed to read videos directory")
        exit(1)
    }

    let videoExtensions = ["mp4", "mov", "m4v", "MP4", "MOV", "M4V"]
    let videos = videoFiles.filter { videoExtensions.contains($0.pathExtension) }.sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !videos.isEmpty else {
        print("✗ No video files found in: \(videosDir.path)")
        print("  Supported formats: MP4, MOV, M4V")
        exit(1)
    }

    print("Videos dir:  \(videosDir.path)")
    print("Output dir:  \(skeletonsDir.path)")
    print("Target fps:  \(targetFPS)")
    print("Found \(videos.count) video file(s)\n")

    let extractor = SkeletonExtractor()
    var successCount = 0
    var failCount = 0

    for (index, videoURL) in videos.enumerated() {
        let filename = videoURL.lastPathComponent
        print("[\(index + 1)/\(videos.count)] \(filename)")

        let outputFilename = videoURL.deletingPathExtension().lastPathComponent + ".json"
        let outputURL = skeletonsDir.appendingPathComponent(outputFilename)

        // Skip already-processed files (resume support)
        if fileManager.fileExists(atPath: outputURL.path) {
            print("  ↷ Skipped (already exists)")
            successCount += 1
            continue
        }

        if let skeleton = extractor.extractSkeletons(from: videoURL, fps: targetFPS) {
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let jsonData = try encoder.encode(skeleton)
                try jsonData.write(to: outputURL)

                let sizeKB = Double(jsonData.count) / 1024.0
                print("  ✓ \(outputFilename)  (\(skeleton.total_frames) frames, \(String(format: "%.1f", sizeKB)) KB)")
                successCount += 1
            } catch {
                print("  ✗ Failed to save JSON: \(error.localizedDescription)")
                failCount += 1
            }
        } else {
            print("  ✗ Failed to extract skeletons")
            failCount += 1
        }
    }

    print("\n" + "=" * 50)
    print("Completed: \(successCount) succeeded, \(failCount) failed")
    print("JSON files saved to: \(skeletonsDir.path)")
}

// Helper for string repetition
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Run main
main()
