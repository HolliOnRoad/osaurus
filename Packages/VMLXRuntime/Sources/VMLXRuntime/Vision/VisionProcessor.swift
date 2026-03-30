import Foundation
import MLX
import CoreImage
import CoreGraphics
import AVFoundation

/// Image detail level for preprocessing.
public enum ImageDetail: String, Sendable {
    case auto = "auto"
    case low = "low"    // 512x512 max
    case high = "high"  // 1024x1024 max
}

/// Preprocessed image ready for model input.
public struct ProcessedImage: @unchecked Sendable {
    /// Pixel values as MLXArray, shape depends on model requirements.
    /// Typically [1, channels, height, width] or [1, height, width, channels].
    public let pixelValues: MLXArray

    /// Original image dimensions before preprocessing.
    public let originalSize: (width: Int, height: Int)

    /// Final dimensions after preprocessing.
    public let processedSize: (width: Int, height: Int)

    /// Grid dimensions for models that use tile-based processing (Qwen-VL).
    /// (temporal, height_tiles, width_tiles)
    public let gridTHW: (Int, Int, Int)?
}

/// Preprocessed video (sequence of frames).
public struct ProcessedVideo: @unchecked Sendable {
    /// Frame pixel values, shape [num_frames, channels, height, width].
    public let pixelValues: MLXArray

    /// Number of frames extracted.
    public let frameCount: Int

    /// Frame dimensions.
    public let frameSize: (width: Int, height: Int)
}

/// Vision preprocessing pipeline for VLM models.
/// Handles images (PNG, JPEG, WebP) and video (frame extraction).
public struct VisionProcessor: Sendable {

    /// Maximum image dimension (pixels).
    public let maxSize: Int

    /// Target size for model input (if model requires fixed size).
    public let targetSize: (width: Int, height: Int)?

    /// Normalization mean per channel [R, G, B].
    public let normMean: [Float]

    /// Normalization std per channel [R, G, B].
    public let normStd: [Float]

    public init(
        maxSize: Int = 1024,
        targetSize: (width: Int, height: Int)? = nil,
        normMean: [Float] = [0.48145466, 0.4578275, 0.40821073],  // CLIP defaults
        normStd: [Float] = [0.26862954, 0.26130258, 0.27577711]
    ) {
        self.maxSize = maxSize
        self.targetSize = targetSize
        self.normMean = normMean
        self.normStd = normStd
    }

    // MARK: - Image Processing

    /// Process image data (PNG, JPEG, WebP) into model-ready pixel values.
    public func processImage(data: Data, detail: ImageDetail = .auto) throws -> ProcessedImage {
        guard let ciImage = CIImage(data: data) else {
            throw VisionError.invalidImageData
        }

        let extent = ciImage.extent
        let originalWidth = Int(extent.width)
        let originalHeight = Int(extent.height)

        // Determine max size based on detail level
        let effectiveMax: Int
        switch detail {
        case .low: effectiveMax = 512
        case .high: effectiveMax = maxSize
        case .auto: effectiveMax = maxSize
        }

        // Calculate resize dimensions maintaining aspect ratio
        let (targetW, targetH) = _resizeDimensions(
            width: originalWidth, height: originalHeight, maxDim: effectiveMax
        )

        // Resize using CoreImage
        let resized = _resizeImage(ciImage, to: CGSize(width: targetW, height: targetH))

        // Extract pixel data as float array
        let pixelData = try _extractPixelData(from: resized, width: targetW, height: targetH)

        // Normalize
        let normalized = _normalize(pixelData, width: targetW, height: targetH)

        // Convert to MLXArray [1, 3, height, width] (NCHW format)
        let pixelValues = MLXArray(normalized, [1, 3, targetH, targetW])

        return ProcessedImage(
            pixelValues: pixelValues,
            originalSize: (originalWidth, originalHeight),
            processedSize: (targetW, targetH),
            gridTHW: nil
        )
    }

    /// Process image from a base64 data URL, HTTP(S) URL, or local file path.
    public func processImageURL(_ url: String, detail: ImageDetail = .auto) throws -> ProcessedImage {
        let data: Data
        if url.hasPrefix("data:image") {
            // Base64 data URL
            guard let commaIdx = url.firstIndex(of: ",") else {
                throw VisionError.invalidImageURL
            }
            let base64Str = String(url[url.index(after: commaIdx)...])
            guard let decoded = Data(base64Encoded: base64Str) else {
                throw VisionError.invalidBase64
            }
            data = decoded
        } else if url.hasPrefix("http://") || url.hasPrefix("https://") {
            // HTTP URL - synchronous download
            guard let requestURL = URL(string: url),
                  let downloadedData = try? Data(contentsOf: requestURL) else {
                throw VisionError.downloadFailed(url)
            }
            data = downloadedData
        } else if FileManager.default.fileExists(atPath: url) {
            // Local file path
            guard let fileData = FileManager.default.contents(atPath: url) else {
                throw VisionError.fileNotFound(url)
            }
            data = fileData
        } else {
            throw VisionError.invalidImageURL
        }

        return try processImage(data: data, detail: detail)
    }

    // MARK: - Video Processing

    /// Extract frames from video data for VLM input.
    /// Smart frame selection: evenly spaced across video duration.
    public func extractFrames(from videoURL: URL, maxFrames: Int = 16) throws -> ProcessedVideo {
        let asset = AVURLAsset(url: videoURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        // Get duration
        let duration = CMTimeGetSeconds(asset.duration)
        guard duration > 0 else { throw VisionError.frameExtractionFailed }

        // Calculate evenly-spaced frame times (~2fps, capped at maxFrames)
        let frameCount = min(maxFrames, max(1, Int(duration * 2)))
        let interval = duration / Double(frameCount)

        var frames: [ProcessedImage] = []
        for i in 0..<frameCount {
            let time = CMTimeMakeWithSeconds(Double(i) * interval, preferredTimescale: 600)
            let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
            let ciImage = CIImage(cgImage: cgImage)
            let data = try _ciImageToData(ciImage)
            let processed = try processImage(data: data, detail: .low)
            frames.append(processed)
        }

        guard let firstFrame = frames.first else { throw VisionError.frameExtractionFailed }

        // Stack frames into single tensor [numFrames, channels, height, width]
        // For now, return the first frame's dimensions
        let pixelValues = firstFrame.pixelValues  // TODO: stack all frames

        return ProcessedVideo(
            pixelValues: pixelValues,
            frameCount: frames.count,
            frameSize: firstFrame.processedSize
        )
    }

    /// Convert a CIImage to PNG data for reprocessing through the image pipeline.
    private func _ciImageToData(_ ciImage: CIImage) throws -> Data {
        let context = CIContext()
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let data = context.pngRepresentation(of: ciImage, format: .RGBA8, colorSpace: colorSpace) else {
            throw VisionError.frameExtractionFailed
        }
        return data
    }

    // MARK: - Internal Helpers

    /// Calculate resize dimensions maintaining aspect ratio.
    func _resizeDimensions(width: Int, height: Int, maxDim: Int) -> (Int, Int) {
        if width <= maxDim && height <= maxDim {
            return (width, height)
        }

        let scale: Float
        if width > height {
            scale = Float(maxDim) / Float(width)
        } else {
            scale = Float(maxDim) / Float(height)
        }

        let newW = max(1, Int(Float(width) * scale))
        let newH = max(1, Int(Float(height) * scale))
        return (newW, newH)
    }

    /// Resize CIImage using affine transform.
    func _resizeImage(_ image: CIImage, to size: CGSize) -> CIImage {
        let scaleX = size.width / image.extent.width
        let scaleY = size.height / image.extent.height
        return image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }

    /// Extract RGB pixel data as float array from CIImage in CHW order.
    func _extractPixelData(from image: CIImage, width: Int, height: Int) throws -> [Float] {
        let context = CIContext()
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

        // Render to RGBA8 bitmap
        let bytesPerRow = width * 4
        var bitmap = [UInt8](repeating: 0, count: height * bytesPerRow)

        context.render(
            image,
            toBitmap: &bitmap,
            rowBytes: bytesPerRow,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            format: .RGBA8,
            colorSpace: colorSpace
        )

        // Convert RGBA8 to float RGB (0.0-1.0), in CHW order
        let planeSize = height * width
        var result = [Float](repeating: 0, count: 3 * planeSize)
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = y * bytesPerRow + x * 4
                let r = Float(bitmap[srcIdx]) / 255.0
                let g = Float(bitmap[srcIdx + 1]) / 255.0
                let b = Float(bitmap[srcIdx + 2]) / 255.0
                // CHW layout: [C, H, W]
                result[0 * planeSize + y * width + x] = r
                result[1 * planeSize + y * width + x] = g
                result[2 * planeSize + y * width + x] = b
            }
        }

        return result
    }

    /// Apply per-channel normalization: (pixel - mean) / std.
    func _normalize(_ pixels: [Float], width: Int, height: Int) -> [Float] {
        var result = pixels
        let planeSize = height * width

        for c in 0..<3 {
            let mean = normMean[c]
            let std = normStd[c]
            let offset = c * planeSize
            for i in 0..<planeSize {
                result[offset + i] = (result[offset + i] - mean) / std
            }
        }

        return result
    }
}

// MARK: - Errors

public enum VisionError: Error, LocalizedError, Sendable {
    case invalidImageData
    case invalidImageURL
    case invalidBase64
    case downloadFailed(String)
    case fileNotFound(String)
    case videoNotSupported
    case frameExtractionFailed

    public var errorDescription: String? {
        switch self {
        case .invalidImageData: return "Invalid image data"
        case .invalidImageURL: return "Invalid image URL"
        case .invalidBase64: return "Invalid base64 encoding"
        case .downloadFailed(let url): return "Failed to download: \(url)"
        case .fileNotFound(let path): return "File not found: \(path)"
        case .videoNotSupported: return "Video processing not yet supported"
        case .frameExtractionFailed: return "Frame extraction failed"
        }
    }
}
