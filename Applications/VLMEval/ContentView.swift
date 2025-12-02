// Copyright 2024 Apple Inc.

import AVKit
import AsyncAlgorithms
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import PhotosUI
import SwiftUI
import Vision

#if os(iOS) || os(visionOS)
    typealias PlatformImage = UIImage
#else
    typealias PlatformImage = NSImage
#endif

let videoSystemPrompt =
    "Focus only on describing the key dramatic action or notable event occurring in this video segment. Skip general context or scene-setting details unless they are crucial to understanding the main action."
let imageSystemPrompt =
    "You are an image understanding model capable of describing the salient features of any image."

struct ContentView: View {

    @State var llm = VLMEvaluator()

    @Environment(DeviceStat.self) private var deviceStat
    
    enum FrameSpecificationType: String, CaseIterable {
        case allFrames = "All Frames"
        case frameNumbers = "Frame Numbers"
        case timestamps = "Timestamps"
    }
    @State private var sceneThreshold: Float = 0.05
    @State private var visionThreshold: Float = 0.5
    @State private var minSceneDuration: Float = 2.0
    @State private var maxSceneDuration: Float = 15.0
    @State private var frameSpecification: String = ""
    @State private var frameSpecificationType: FrameSpecificationType = .allFrames

    @State private var selectedImage: PlatformImage? = nil {
        didSet {
            if selectedImage != nil {
                selectedVideoURL = nil
                player = nil
            }
        }
    }
    @State private var selectedVideoURL: URL? {
        didSet {
            if let selectedVideoURL {
                player = AVPlayer(url: selectedVideoURL)
                selectedImage = nil
            }
        }
    }
    @State private var showingImagePicker = false
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var player: AVPlayer? = nil

    private var currentImageURL: URL? {
        selectedImage == nil && selectedVideoURL == nil
            ? URL(
                string:
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
            ) : nil
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }

                VStack {
                    if let player {
                        VideoPlayer(player: player)
                            .frame(height: 300)
                            .cornerRadius(12)
                    } else if let selectedImage {
                        Group {
                            #if os(iOS) || os(visionOS)
                                Image(uiImage: selectedImage)
                                    .resizable()
                            #else
                                Image(nsImage: selectedImage)
                                    .resizable()
                            #endif
                        }
                        .scaledToFit()
                        .cornerRadius(12)
                        .frame(height: 300)
                    } else if let imageURL = currentImageURL {
                        AsyncImage(url: imageURL) { phase in
                            switch phase {
                            case .empty:
                                ProgressView()
                            case .success(let image):
                                image
                                    .resizable()
                                    .scaledToFit()
                                    .cornerRadius(12)
                                    .frame(height: 200)
                            case .failure:
                                Image(systemName: "photo.badge.exclamationmark")
                            @unknown default:
                                EmptyView()
                            }
                        }
                    }

                    HStack {
                        #if os(iOS) || os(visionOS)
                            PhotosPicker(
                                selection: $selectedItem,
                                matching: PHPickerFilter.any(of: [
                                    PHPickerFilter.images, PHPickerFilter.videos,
                                ])
                            ) {
                                Label("Select Image/Video", systemImage: "photo.badge.plus")
                            }
                            .onChange(of: selectedItem) {
                                Task {
                                    if let video = try? await selectedItem?.loadTransferable(
                                        type: TransferableVideo.self)
                                    {
                                        selectedVideoURL = video.url
                                    } else if let data = try? await selectedItem?.loadTransferable(
                                        type: Data.self)
                                    {
                                        selectedImage = PlatformImage(data: data)
                                    }
                                }
                            }
                        #else
                            Button("Select Image/Video") {
                                showingImagePicker = true
                            }
                            .fileImporter(
                                isPresented: $showingImagePicker,
                                allowedContentTypes: [.image, .movie]
                            ) { result in
                                switch result {
                                case .success(let file):
                                    Task { @MainActor in
                                        do {
                                            let data = try loadData(from: file)
                                            if let image = PlatformImage(data: data) {
                                                selectedImage = image
                                            } else if let fileType = UTType(
                                                filenameExtension: file.pathExtension),
                                                fileType.conforms(to: .movie)
                                            {
                                                if let sandboxURL = try? loadVideoToSandbox(
                                                    from: file)
                                                {
                                                    selectedVideoURL = sandboxURL
                                                }
                                            } else {
                                                print("Failed to create image from data")
                                            }
                                        } catch {
                                            print(
                                                "Failed to load image: \(error.localizedDescription)"
                                            )
                                        }
                                    }
                                case .failure(let error):
                                    print(error.localizedDescription)
                                }
                            }
                        #endif

                        if selectedImage != nil {
                            Button("Clear", role: .destructive) {
                                selectedImage = nil
                                selectedItem = nil
                            }
                        }
                    }
                }
                .padding()

                HStack {
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                }
            }

            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Text(llm.output)
                        .textSelection(.enabled)
                        .onChange(of: llm.output) { _, _ in
                            sp.scrollTo("bottom")
                        }

                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }
            .frame(minHeight: 200)

            HStack {
                TextField("prompt", text: Bindable(llm).prompt)
                    .onSubmit(generate)
                    .disabled(llm.running)
                    #if os(visionOS)
                        .textFieldStyle(.roundedBorder)
                    #endif
                Button(llm.running ? "stop" : "generate", action: llm.running ? cancel : generate)
                Button("Extract Patches", action: extractPatches)
                    .disabled(llm.running || (selectedImage == nil && currentImageURL == nil))
                Button("Mean Pool", action: meanPool)
                    .disabled(llm.running || (selectedImage == nil && currentImageURL == nil))
                Button("Compare Similarity", action: compareSimilarity)
                    .disabled(llm.running || (selectedImage == nil && currentImageURL == nil))
                Button("Video Patches", action: extractVideoPatches)
                    .disabled(llm.running || selectedVideoURL == nil)
                Button("Video Mean Pool", action: videoMeanPool)
                    .disabled(llm.running || selectedVideoURL == nil)
                Button("Video Similarity", action: calculateVideoSimilarity)
                    .disabled(llm.running || selectedVideoURL == nil)
                Button("Scene Detection", action: detectSceneChanges)
                    .disabled(llm.running || selectedVideoURL == nil)
                Button("Vision Feature Prints", action: extractVisionFeaturePrints)
                    .disabled(llm.running || selectedVideoURL == nil)
                Button("Vision Distances", action: calculateVisionDistances)
                    .disabled(llm.running || selectedVideoURL == nil)
                Button("Vision Scene Detection", action: detectVisionSceneChanges)
                    .disabled(llm.running || selectedVideoURL == nil)
            }
            
            if selectedVideoURL != nil {
                VStack(spacing: 16) {
                    Text("Scene Change Detection Settings")
                        .font(.headline)
                        .padding(.top)
                    
                    VStack(alignment: .center, spacing: 12) {
                        Text("MLX Threshold: \(String(format: "%.2f", sceneThreshold))")
                            .font(.title2)
                            .fontWeight(.medium)
                        
                        HStack(spacing: 20) {
                            Button(action: {
                                if sceneThreshold > 0.01 {
                                    sceneThreshold = max(0.01, sceneThreshold - 0.01)
                                }
                            }) {
                                Image(systemName: "minus.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.blue)
                            }
                            .disabled(llm.running || sceneThreshold <= 0.01)
                            
                            VStack(spacing: 4) {
                                Text("Current Value")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.2f", sceneThreshold))
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .monospacedDigit()
                            }
                            
                            Button(action: {
                                if sceneThreshold < 0.5 {
                                    sceneThreshold = min(0.5, sceneThreshold + 0.01)
                                }
                            }) {
                                Image(systemName: "plus.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.blue)
                            }
                            .disabled(llm.running || sceneThreshold >= 0.5)
                        }
                        
                        VStack(spacing: 4) {
                            Text("Range: 0.01 - 0.50")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Step: 0.01")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Divider()
                            .padding(.vertical, 8)
                        
                        Text("Vision Threshold: \(String(format: "%.2f", visionThreshold))")
                            .font(.title2)
                            .fontWeight(.medium)
                        
                        HStack(spacing: 20) {
                            Button(action: {
                                if visionThreshold > 0.01 {
                                    visionThreshold = max(0.01, visionThreshold - 0.01)
                                }
                            }) {
                                Image(systemName: "minus.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.orange)
                            }
                            .disabled(llm.running || visionThreshold <= 0.01)
                            
                            VStack(spacing: 4) {
                                Text("Current Value")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.2f", visionThreshold))
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .monospacedDigit()
                            }
                            
                            Button(action: {
                                if visionThreshold < 1.0 {
                                    visionThreshold = min(1.0, visionThreshold + 0.01)
                                }
                            }) {
                                Image(systemName: "plus.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.orange)
                            }
                            .disabled(llm.running || visionThreshold >= 1.0)
                        }
                        
                        VStack(spacing: 4) {
                            Text("Range: 0.01 - 1.00")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Step: 0.01")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Divider()
                            .padding(.vertical, 8)
                        
                        Text("Min Scene Duration: \(String(format: "%.1f", minSceneDuration))s")
                            .font(.title2)
                            .fontWeight(.medium)
                        
                        HStack(spacing: 20) {
                            Button(action: {
                                if minSceneDuration > 0.5 {
                                    minSceneDuration = max(0.5, minSceneDuration - 0.5)
                                }
                            }) {
                                Image(systemName: "minus.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.blue)
                            }
                            .disabled(llm.running || minSceneDuration <= 0.5)
                            
                            VStack(spacing: 4) {
                                Text("Current Value")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.1f", minSceneDuration))
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .monospacedDigit()
                            }
                            
                            Button(action: {
                                if minSceneDuration < 10.0 {
                                    minSceneDuration = min(10.0, minSceneDuration + 0.5)
                                }
                            }) {
                                Image(systemName: "plus.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.blue)
                            }
                            .disabled(llm.running || minSceneDuration >= 10.0)
                        }
                        
                        VStack(spacing: 4) {
                            Text("Range: 0.5s - 10.0s")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Step: 0.5s")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Divider()
                            .padding(.vertical, 8)
                        
                        Text("Max Scene Duration: \(String(format: "%.1f", maxSceneDuration))s")
                            .font(.title2)
                            .fontWeight(.medium)
                        
                        HStack(spacing: 20) {
                            Button(action: {
                                if maxSceneDuration > 5.0 {
                                    maxSceneDuration = max(5.0, maxSceneDuration - 1.0)
                                }
                            }) {
                                Image(systemName: "minus.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.blue)
                            }
                            .disabled(llm.running || maxSceneDuration <= 5.0)
                            
                            VStack(spacing: 4) {
                                Text("Current Value")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.1f", maxSceneDuration))
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .monospacedDigit()
                            }
                            
                            Button(action: {
                                if maxSceneDuration < 60.0 {
                                    maxSceneDuration = min(60.0, maxSceneDuration + 1.0)
                                }
                            }) {
                                Image(systemName: "plus.circle.fill")
                                    .font(.title)
                                    .foregroundColor(.blue)
                            }
                            .disabled(llm.running || maxSceneDuration >= 60.0)
                        }
                        
                        VStack(spacing: 4) {
                            Text("Range: 5.0s - 60.0s")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Step: 1.0s")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Divider()
                            .padding(.vertical, 8)
                        
                        Text("Frame Selection")
                            .font(.title2)
                            .fontWeight(.medium)
                        
                        Picker("Frame Specification Type", selection: $frameSpecificationType) {
                            ForEach(FrameSpecificationType.allCases, id: \.self) { type in
                                Text(type.rawValue).tag(type)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .disabled(llm.running)
                        
                        if frameSpecificationType != .allFrames {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Frame Specification:")
                                    .font(.headline)
                                
                                TextField("Enter values (comma-separated)", text: $frameSpecification)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                    .disabled(llm.running)
                                
                                Text(frameSpecificationType == .frameNumbers ? 
                                     "Example: 0, 10, 20, 30 (frame numbers)" : 
                                     "Example: 0.0, 5.0, 10.0, 15.0 (timestamps in seconds)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.top, 8)
                        }
                    }
                    .padding()
                    .background(Color.secondary.opacity(0.1))
                    .cornerRadius(12)
                    .padding(.horizontal)
                }
            }
        }
        .onAppear {
            selectedVideoURL = URL(
                string:
                    "https://videos.pexels.com/video-files/4066325/4066325-uhd_2560_1440_24fps.mp4")!
        }
        #if os(visionOS)
            .padding(40)
        #else
            .padding()
        #endif
        .toolbar {
            ToolbarItem {
                Label(
                    "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))",
                    systemImage: "info.circle.fill"
                )
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
                .help(
                    Text(
                        """
                        Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                        Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                        Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                        """
                    )
                )
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        copyToClipboard(llm.output)
                    }
                } label: {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(llm.output == "")
                .labelStyle(.titleAndIcon)
            }
        }
        .task {
            _ = try? await llm.load()
        }
        }
    }

    private func generate() {
        Task {
            if let selectedImage = selectedImage {
                #if os(iOS) || os(visionOS)
                    let ciImage = CIImage(image: selectedImage)
                    llm.generate(image: ciImage ?? CIImage(), videoURL: nil)
                #else
                    if let cgImage = selectedImage.cgImage(
                        forProposedRect: nil, context: nil, hints: nil)
                    {
                        let ciImage = CIImage(cgImage: cgImage)
                        llm.generate(image: ciImage, videoURL: nil)
                    }
                #endif
            } else if let imageURL = currentImageURL {
                do {
                    let (data, _) = try await URLSession.shared.data(from: imageURL)
                    if let ciImage = CIImage(data: data) {
                        llm.generate(image: ciImage, videoURL: nil)
                    }
                } catch {
                    print("Failed to load image: \(error.localizedDescription)")
                }
            } else {
                if let videoURL = selectedVideoURL {
                    let frameSpec = createFrameSpecification()
                    llm.generate(image: nil, videoURL: videoURL, frameSpecification: frameSpec)
                }
            }
        }
    }

    private func cancel() {
        llm.cancelGeneration()
    }
    
    private func extractPatches() {
        Task {
            if let selectedImage = selectedImage {
                #if os(iOS) || os(visionOS)
                    let ciImage = CIImage(image: selectedImage)
                    llm.extractPatches(image: ciImage ?? CIImage())
                #else
                    if let cgImage = selectedImage.cgImage(
                        forProposedRect: nil, context: nil, hints: nil)
                    {
                        let ciImage = CIImage(cgImage: cgImage)
                        llm.extractPatches(image: ciImage)
                    }
                #endif
            } else if let imageURL = currentImageURL {
                do {
                    let (data, _) = try await URLSession.shared.data(from: imageURL)
                    if let ciImage = CIImage(data: data) {
                        llm.extractPatches(image: ciImage)
                    }
                } catch {
                    print("Failed to load image: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func meanPool() {
        Task {
            if let selectedImage = selectedImage {
                #if os(iOS) || os(visionOS)
                    let ciImage = CIImage(image: selectedImage)
                    llm.meanPool(image: ciImage ?? CIImage())
                #else
                    if let cgImage = selectedImage.cgImage(
                        forProposedRect: nil, context: nil, hints: nil)
                    {
                        let ciImage = CIImage(cgImage: cgImage)
                        llm.meanPool(image: ciImage)
                    }
                #endif
            } else if let imageURL = currentImageURL {
                do {
                    let (data, _) = try await URLSession.shared.data(from: imageURL)
                    if let ciImage = CIImage(data: data) {
                        llm.meanPool(image: ciImage)
                    }
                } catch {
                    print("Failed to load image: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func compareSimilarity() {
        Task {
            if let selectedImage = selectedImage {
                #if os(iOS) || os(visionOS)
                    let ciImage = CIImage(image: selectedImage)
                    llm.compareSimilarity(image: ciImage ?? CIImage())
                #else
                    if let cgImage = selectedImage.cgImage(
                        forProposedRect: nil, context: nil, hints: nil)
                    {
                        let ciImage = CIImage(cgImage: cgImage)
                        llm.compareSimilarity(image: ciImage)
                    }
                #endif
            } else if let imageURL = currentImageURL {
                do {
                    let (data, _) = try await URLSession.shared.data(from: imageURL)
                    if let ciImage = CIImage(data: data) {
                        llm.compareSimilarity(image: ciImage)
                    }
                } catch {
                    print("Failed to load image: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func extractVideoPatches() {
        Task {
            if let videoURL = selectedVideoURL {
                let frameSpec = createFrameSpecification()
                llm.extractVideoPatches(videoURL: videoURL, frameSpecification: frameSpec)
            }
        }
    }
    
    private func videoMeanPool() {
        Task {
            if let videoURL = selectedVideoURL {
                let frameSpec = createFrameSpecification()
                llm.videoMeanPool(videoURL: videoURL, frameSpecification: frameSpec)
            }
        }
    }
    
    private func calculateVideoSimilarity() {
        Task {
            if let videoURL = selectedVideoURL {
                let frameSpec = createFrameSpecification()
                llm.calculateVideoSimilarity(videoURL: videoURL, frameSpecification: frameSpec)
            }
        }
    }
    
    private func detectSceneChanges() {
        Task {
            if let videoURL = selectedVideoURL {
                llm.detectSceneChanges(videoURL: videoURL, threshold: sceneThreshold, minSceneDuration: minSceneDuration, maxSceneDuration: maxSceneDuration)
            }
        }
    }
    
    private func extractVisionFeaturePrints() {
        Task {
            if let videoURL = selectedVideoURL {
                let frameSpec = createFrameSpecification()
                llm.extractVisionFeaturePrints(videoURL: videoURL, frameSpecification: frameSpec)
            }
        }
    }
    
    private func calculateVisionDistances() {
        Task {
            if let videoURL = selectedVideoURL {
                let frameSpec = createFrameSpecification()
                llm.calculateVisionDistances(videoURL: videoURL, frameSpecification: frameSpec)
            }
        }
    }
    
    private func detectVisionSceneChanges() {
        Task {
            if let videoURL = selectedVideoURL {
                llm.detectVisionSceneChanges(videoURL: videoURL, threshold: visionThreshold, minSceneDuration: minSceneDuration, maxSceneDuration: maxSceneDuration)
            }
        }
    }
    
    private func createFrameSpecification() -> FrameSpecification {
        switch frameSpecificationType {
        case .allFrames:
            return .allFrames
        case .frameNumbers:
            let numbers = frameSpecification
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            return .frameNumbers(numbers)
        case .timestamps:
            let timestamps = frameSpecification
                .split(separator: ",")
                .compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            return .timestamps(timestamps)
        }
    }
    


    #if os(macOS)
        private func loadData(from url: URL) throws -> Data {
            guard url.startAccessingSecurityScopedResource() else {
                throw NSError(
                    domain: "FileAccess", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to access the file."])
            }
            defer { url.stopAccessingSecurityScopedResource() }
            return try Data(contentsOf: url)
        }

        private func loadVideoToSandbox(from url: URL) throws -> URL {
            guard url.startAccessingSecurityScopedResource() else {
                throw NSError(
                    domain: "FileAccess", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to access the file."])
            }
            defer { url.stopAccessingSecurityScopedResource() }
            let sandboxURL = try SandboxFileTransfer.transferFileToTemp(from: url)
            return sandboxURL
        }
    #endif

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }
}

@Observable
@MainActor
class VLMEvaluator {

    var running = false

    var prompt = ""
    var output = ""
    var modelInfo = ""
    var stat = ""
    
    let visionProcessor = VisionProcessor()
    
    private func convertToQwen2VLFrameSpecification(_ frameSpec: FrameSpecification) -> Qwen2VL.FrameSpecification {
        switch frameSpec {
        case .allFrames:
            return .allFrames
        case .frameNumbers(let numbers):
            return .frameNumbers(numbers)
        case .timestamps(let times):
            return .timestamps(times)
        }
    }
    
    private func convertToSmolVLM2FrameSpecification(_ frameSpec: FrameSpecification) -> MLXVLM.FrameSpecification {
        switch frameSpec {
        case .allFrames:
            return .allFrames
        case .frameNumbers(let numbers):
            return .frameNumbers(numbers)
        case .timestamps(let times):
            return .timestamps(times)
        }
    }
    
    private func convertToQwen25VLFrameSpecification(_ frameSpec: FrameSpecification) -> Qwen25VL.FrameSpecification {
        switch frameSpec {
        case .allFrames:
            return .allFrames
        case .frameNumbers(let numbers):
            return .frameNumbers(numbers)
        case .timestamps(let times):
            return .timestamps(times)
        }
    }
    
    private func convertToQwen3VLFrameSpecification(_ frameSpec: FrameSpecification) -> Qwen3VL.FrameSpecification {
        switch frameSpec {
        case .allFrames:
            return .allFrames
        case .frameNumbers(let numbers):
            return .frameNumbers(numbers)
        case .timestamps(let times):
            return .timestamps(times)
        }
    }

    /// This controls which model loads. `smolvlm` is very small even unquantized, so it will fit on
    /// more devices.
//    let modelConfiguration = VLMRegistry.smolvlm
//    let modelConfiguration = VLMRegistry.qwen2_5VL3BInstruct4Bit
//    let modelConfiguration = VLMRegistry.qwen2VL2BInstruct4Bit
    let modelConfiguration = VLMRegistry.qwen3VL2BInstruct4Bit

    /// parameters controlling the output – use values appropriate for the model selected above
    let generateParameters = MLXLMCommon.GenerateParameters(
        maxTokens: 800, temperature: 0.7, topP: 0.9)
    let updateInterval = Duration.seconds(0.25)

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) { [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }

            // Modify the processor configuration to set maxFrames and FPS
            await modelContainer.update { context in
                if let qwen2vlProcessor = context.processor as? Qwen2VLProcessor {
                    // Set maxFrames for video processing
                    qwen2vlProcessor.config.maxFrames = 32
                    // Set FPS for video sampling
                    qwen2vlProcessor.config.fps = 1.0
                } else if let qwen25vlProcessor = context.processor as? Qwen25VLProcessor {
                    // Set maxFrames for video processing
                    qwen25vlProcessor.config.maxFrames = 32
                    // Set FPS for video sampling
                    qwen25vlProcessor.config.fps = 1.0
                } else if let qwen3vlProcessor = context.processor as? Qwen3VLProcessor {
                    // Set maxFrames for video processing
                    qwen3vlProcessor.config.maxFrames = 32
                    // Set FPS for video sampling
                    qwen3vlProcessor.config.fps = 1.0
                } else if let smolVLMProcessor = context.processor as? SmolVLMProcessor {
                    // Set maxFrames for video processing
                    smolVLMProcessor.config.maxFrames = 32
                    // Set FPS for video sampling
                    smolVLMProcessor.config.fps = 1.0
                }
            }

            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }

            self.prompt = modelConfiguration.defaultPrompt
            self.modelInfo = "Loaded \(modelConfiguration.id). Weights: \(numParams / (1024*1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    private func generate(prompt: String, image: CIImage?, videoURL: URL?, video: AVAsset?, frameSpecification: FrameSpecification = .allFrames) async {

        self.output = ""

        do {
            let modelContainer = try await load()

            // Set pixel limits based on media type
            await modelContainer.update { context in
                if let qwen2vlProcessor = context.processor as? Qwen2VLProcessor {
                    if (videoURL != nil) || (video != nil) {
                        // For videos: lower pixel limits
                        qwen2vlProcessor.config.maxPixels = 128 * 28 * 28  // 200,704
                        qwen2vlProcessor.config.minPixels = 64 * 28 * 28   // 50,176
                    } else if image != nil {
                        // For images: higher pixel limits
                        qwen2vlProcessor.config.maxPixels = 512 * 28 * 28  // 401,408
                        qwen2vlProcessor.config.minPixels = 128 * 28 * 28  // 200,704
                    }
                } else if let qwen25vlProcessor = context.processor as? Qwen25VLProcessor {
                    if (videoURL != nil) || (video != nil) {
                        // For videos: lower pixel limits
                        qwen25vlProcessor.config.maxPixels = 128 * 28 * 28  // 200,704
                        qwen25vlProcessor.config.minPixels = 64 * 28 * 28   // 50,176
                    } else if image != nil {
                        // For images: higher pixel limits
                        qwen25vlProcessor.config.maxPixels = 512 * 28 * 28  // 401,408
                        qwen25vlProcessor.config.minPixels = 128 * 28 * 28  // 200,704
                    }
                } else if let qwen3vlProcessor = context.processor as? Qwen3VLProcessor {
                    if (videoURL != nil) || (video != nil) {
                        // For videos: lower pixel limits
                        qwen3vlProcessor.config.maxPixels = 128 * 28 * 28  // 200,704
                        qwen3vlProcessor.config.minPixels = 64 * 28 * 28   // 50,176
                    } else if image != nil {
                        // For images: higher pixel limits
                        qwen3vlProcessor.config.maxPixels = 512 * 28 * 28  // 401,408
                        qwen3vlProcessor.config.minPixels = 128 * 28 * 28  // 200,704
                    }
                }
            }

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
            
            // Show frame specification if video is being processed
            if videoURL != nil {
                self.output = "Processing video with frame specification: \(frameSpecification)\n\n"
            }

            try await modelContainer.perform { (context: ModelContext) -> Void in
                let images: [UserInput.Image] = if let image { [.ciImage(image)] } else { [] }
                let videos: [UserInput.Video] = if let videoURL { [.url(videoURL)] } else { [] }

                let systemPrompt =
                    if !videos.isEmpty {
                        videoSystemPrompt
                    } else if !images.isEmpty {
                        imageSystemPrompt
                    } else { "You are a helpful assistant." }

                let chat: [Chat.Message] = [
                    .system(systemPrompt),
                    .user(prompt, images: images, videos: videos),
                ]

                var userInput = UserInput(chat: chat)
//                userInput.processing.resize = .init(width: 448, height: 448)

                let lmInput: LMInput
                if let qwen2vlProcessor = context.processor as? Qwen2VLProcessor, videoURL != nil {
                    // Use frame specification for video processing with Qwen2VL
                    lmInput = try await qwen2vlProcessor.prepareWithFrameSpecification(input: userInput, frameSpecification: convertToQwen2VLFrameSpecification(frameSpecification))
                } else if let qwen25vlProcessor = context.processor as? Qwen25VLProcessor, videoURL != nil {
                    // Use frame specification for video processing with Qwen25VL
                    lmInput = try await qwen25vlProcessor.prepareWithFrameSpecification(input: userInput, frameSpecification: convertToQwen25VLFrameSpecification(frameSpecification))
                } else if let qwen3vlProcessor = context.processor as? Qwen3VLProcessor, videoURL != nil {
                    // Use frame specification for video processing with Qwen3VL
                    lmInput = try await qwen3vlProcessor.prepareWithFrameSpecification(input: userInput, frameSpecification: convertToQwen3VLFrameSpecification(frameSpecification))
                } else if let smolVLMProcessor = context.processor as? SmolVLMProcessor, videoURL != nil {
                    // Use frame specification for video processing with SmolVLM2
                    lmInput = try await smolVLMProcessor.prepareWithFrameSpecification(input: userInput, frameSpecification: convertToSmolVLM2FrameSpecification(frameSpecification))
                } else {
                    // Use regular prepare for images or no media
                    lmInput = try await context.processor.prepare(input: userInput)
                }

                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: generateParameters, context: context)
                // generate and output in batches
                for await batch in stream._throttle(
                    for: updateInterval, reducing: Generation.collect)
                {
                    let output = batch.compactMap { $0.chunk }.joined(separator: "")
                    if !output.isEmpty {
                        Task { @MainActor [output] in
                            self.output += output
                        }
                    }

                    if let completion = batch.compactMap({ $0.info }).first {
                        Task { @MainActor in
                            self.stat = "\(completion.tokensPerSecond) tokens/s"
                        }
                    }
                }
            }
        } catch {
            output = "Failed: \(error)"
        }
    }

    func generate(image: CIImage?, videoURL: URL?, frameSpecification: FrameSpecification = .allFrames) {
        guard !running else { return }
        let currentPrompt = prompt
        prompt = ""
        generationTask = Task {
            running = true
            await generate(prompt: currentPrompt, image: image, videoURL: videoURL, video: nil, frameSpecification: frameSpecification)
            running = false
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        running = false
    }
    
    func extractPatches(image: CIImage) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await extractPatchesAsync(image: image)
            running = false
        }
    }
    
    private func extractPatchesAsync(image: CIImage) async {
        self.output = ""
        
        do {
            let modelContainer = try await load()
            
            let result =             try await modelContainer.perform { (context: ModelContext) -> String in
                guard let qwen2VL = context.model as? Qwen2VL else {
                    return "Error: Model is not Qwen2VL"
                }
                
                guard let processorConfig = (context.processor as? Qwen2VLProcessor)?.config else {
                    return "Error: Processor is not Qwen2VLProcessor"
                }
                
                let patchEmbeddings = try qwen2VL.extractPatchEmbeddings(from: image, processorConfig: processorConfig)
                return "Successfully extracted patch embeddings with shape: \(patchEmbeddings.shape)"
            }
            
            self.output = result
        } catch {
            self.output = "Failed to extract patches: \(error)"
        }
    }
    
    func meanPool(image: CIImage) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await meanPoolAsync(image: image)
            running = false
        }
    }
    
    private func meanPoolAsync(image: CIImage) async {
        self.output = ""
        
        do {
            let modelContainer = try await load()
            
            let result = try await modelContainer.perform { (context: ModelContext) -> String in
                guard let qwen2VL = context.model as? Qwen2VL else {
                    return "Error: Model is not Qwen2VL"
                }
                
                guard let processorConfig = (context.processor as? Qwen2VLProcessor)?.config else {
                    return "Error: Processor is not Qwen2VLProcessor"
                }
                
                let pooledEmbeddings = try qwen2VL.extractAndPoolEmbeddings(from: image, processorConfig: processorConfig)
                return "Successfully extracted and mean-pooled embeddings with shape: \(pooledEmbeddings.shape)"
            }
            
            self.output = result
        } catch {
            self.output = "Failed to mean pool: \(error)"
        }
    }
    
    func compareSimilarity(image: CIImage) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await compareSimilarityAsync(image: image)
            running = false
        }
    }
    
    private func compareSimilarityAsync(image: CIImage) async {
        self.output = ""
        
        do {
            let modelContainer = try await load()
            
            let result = try await modelContainer.perform { (context: ModelContext) -> String in
                guard let qwen2VL = context.model as? Qwen2VL else {
                    return "Error: Model is not Qwen2VL"
                }
                
                guard let processorConfig = (context.processor as? Qwen2VLProcessor)?.config else {
                    return "Error: Processor is not Qwen2VLProcessor"
                }
                
                // Get embedding for the current image
                let currentEmbedding = try qwen2VL.extractAndPoolEmbeddings(from: image, processorConfig: processorConfig)
                
                // Create a reference image (you can modify this to use a different reference)
                // For now, we'll compare with the same image to test the function
                let referenceEmbedding = try qwen2VL.extractAndPoolEmbeddings(from: image, processorConfig: processorConfig)
                
                // Calculate cosine distance
                let distance = qwen2VL.cosineDistance(currentEmbedding, referenceEmbedding)
                
                return "Cosine distance: \(distance)\nCurrent embedding shape: \(currentEmbedding.shape)\nReference embedding shape: \(referenceEmbedding.shape)"
            }
            
            self.output = result
        } catch {
            self.output = "Failed to compare similarity: \(error)"
        }
    }
    
    func extractVideoPatches(videoURL: URL, frameSpecification: FrameSpecification = .allFrames) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await extractVideoPatchesAsync(videoURL: videoURL, frameSpecification: frameSpecification)
            running = false
        }
    }
    
    private func extractVideoPatchesAsync(videoURL: URL, frameSpecification: FrameSpecification) async {
        self.output = ""
        
        do {
            let modelContainer = try await load()
            
            let result = try await modelContainer.perform { (context: ModelContext) -> String in
                if let qwen2VL = context.model as? Qwen2VL {
                    guard let processorConfig = (context.processor as? Qwen2VLProcessor)?.config else {
                        return "Error: Processor is not Qwen2VLProcessor"
                    }
                    
                    let frameEmbeddings = try await qwen2VL.extractVideoPatchEmbeddings(from: videoURL, processorConfig: processorConfig)
                    
                    var output = "Frame specification: \(frameSpecification)\n"
                    output += "Successfully extracted patch embeddings for \(frameEmbeddings.count) frames:\n"
                    for (index, embedding) in frameEmbeddings.enumerated() {
                        output += "Frame \(index + 1): shape \(embedding.shape), size \(embedding.size)\n"
                    }
                    
                    return output
                } else if let qwen3VL = context.model as? Qwen3VL {
                    guard let processorConfig = (context.processor as? Qwen3VLProcessor)?.config else {
                        return "Error: Processor is not Qwen3VLProcessor"
                    }
                    
                    let frameEmbeddings = try await qwen3VL.extractVideoPatchEmbeddings(from: videoURL, processorConfig: processorConfig)
                    
                    var output = "Frame specification: \(frameSpecification)\n"
                    output += "Successfully extracted patch embeddings for \(frameEmbeddings.count) frames:\n"
                    for (index, embedding) in frameEmbeddings.enumerated() {
                        output += "Frame \(index + 1): shape \(embedding.shape), size \(embedding.size)\n"
                    }
                    
                    return output
                } else if let qwen25VL = context.model as? Qwen25VL {
                    return "Video patch extraction not yet implemented for Qwen25VL"
                } else {
                    return "Error: Model is not Qwen2VL, Qwen3VL, or Qwen25VL"
                }
            }
            
            self.output = result
        } catch {
            self.output = "Failed to extract video patches: \(error)"
        }
    }
    
    func videoMeanPool(videoURL: URL, frameSpecification: FrameSpecification = .allFrames) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await videoMeanPoolAsync(videoURL: videoURL, frameSpecification: frameSpecification)
            running = false
        }
    }
    
    private func videoMeanPoolAsync(videoURL: URL, frameSpecification: FrameSpecification) async {
        self.output = ""
        
        do {
            let modelContainer = try await load()
            
            let result = try await modelContainer.perform { (context: ModelContext) -> String in
                if let qwen2VL = context.model as? Qwen2VL {
                    guard let processorConfig = (context.processor as? Qwen2VLProcessor)?.config else {
                        return "Error: Processor is not Qwen2VLProcessor"
                    }
                    
                    let frameEmbeddings = try await qwen2VL.extractAndPoolVideoEmbeddings(from: videoURL, frameSpecification: convertToQwen2VLFrameSpecification(frameSpecification), processorConfig: processorConfig)
                    
                    var output = "Frame specification: \(frameSpecification)\n"
                    output += "Successfully extracted and mean-pooled embeddings for \(frameEmbeddings.count) frames:\n"
                    for (index, embedding) in frameEmbeddings.enumerated() {
                        output += "Frame \(index + 1): shape \(embedding.shape), size \(embedding.size)\n"
                    }
                    
                    return output
                } else if let qwen3VL = context.model as? Qwen3VL {
                    guard let processorConfig = (context.processor as? Qwen3VLProcessor)?.config else {
                        return "Error: Processor is not Qwen3VLProcessor"
                    }
                    
                    let frameEmbeddings = try await qwen3VL.extractAndPoolVideoEmbeddings(from: videoURL, frameSpecification: convertToQwen3VLFrameSpecification(frameSpecification), processorConfig: processorConfig)
                    
                    var output = "Frame specification: \(frameSpecification)\n"
                    output += "Successfully extracted and mean-pooled embeddings for \(frameEmbeddings.count) frames:\n"
                    for (index, embedding) in frameEmbeddings.enumerated() {
                        output += "Frame \(index + 1): shape \(embedding.shape), size \(embedding.size)\n"
                    }
                    
                    return output
                } else if let qwen25VL = context.model as? Qwen25VL {
                    return "Video mean pooling not yet implemented for Qwen25VL"
                } else {
                    return "Error: Model is not Qwen2VL, Qwen3VL, or Qwen25VL"
                }
            }
            
            self.output = result
        } catch {
            self.output = "Failed to video mean pool: \(error)"
        }
    }
    
    func calculateVideoSimilarity(videoURL: URL, frameSpecification: FrameSpecification = .allFrames) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await calculateVideoSimilarityAsync(videoURL: videoURL, frameSpecification: frameSpecification)
            running = false
        }
    }
    
    private func calculateVideoSimilarityAsync(videoURL: URL, frameSpecification: FrameSpecification) async {
        self.output = ""
        
        do {
            let modelContainer = try await load()
            
            let result = try await modelContainer.perform { (context: ModelContext) -> String in
                if let qwen2VL = context.model as? Qwen2VL {
                    guard let processorConfig = (context.processor as? Qwen2VLProcessor)?.config else {
                        return "Error: Processor is not Qwen2VLProcessor"
                    }
                    
                    let distances = try await qwen2VL.calculateVideoFrameDistances(from: videoURL, frameSpecification: convertToQwen2VLFrameSpecification(frameSpecification), processorConfig: processorConfig)
                    
                    var output = "Frame specification: \(frameSpecification)\n"
                    output += "Cosine distances to reference frame (Frame 1):\n"
                    for (index, distance) in distances.enumerated() {
                        output += "Frame \(index + 1): \(String(format: "%.4f", distance))\n"
                    }
                    
                    // Add summary statistics
                    let minDistance = distances.min() ?? 0.0
                    let maxDistance = distances.max() ?? 0.0
                    let avgDistance = distances.reduce(0, +) / Float(distances.count)
                    
                    output += "\nSummary:\n"
                    output += "Min distance: \(String(format: "%.4f", minDistance))\n"
                    output += "Max distance: \(String(format: "%.4f", maxDistance))\n"
                    output += "Average distance: \(String(format: "%.4f", avgDistance))\n"
                    output += "Total frames: \(distances.count)\n"
                    
                    return output
                } else if let qwen3VL = context.model as? Qwen3VL {
                    guard let processorConfig = (context.processor as? Qwen3VLProcessor)?.config else {
                        return "Error: Processor is not Qwen3VLProcessor"
                    }
                    
                    let distances = try await qwen3VL.calculateVideoFrameDistances(from: videoURL, frameSpecification: convertToQwen3VLFrameSpecification(frameSpecification), processorConfig: processorConfig)
                    
                    var output = "Frame specification: \(frameSpecification)\n"
                    output += "Cosine distances to reference frame (Frame 1):\n"
                    for (index, distance) in distances.enumerated() {
                        output += "Frame \(index + 1): \(String(format: "%.4f", distance))\n"
                    }
                    
                    // Add summary statistics
                    let minDistance = distances.min() ?? 0.0
                    let maxDistance = distances.max() ?? 0.0
                    let avgDistance = distances.reduce(0, +) / Float(distances.count)
                    
                    output += "\nSummary:\n"
                    output += "Min distance: \(String(format: "%.4f", minDistance))\n"
                    output += "Max distance: \(String(format: "%.4f", maxDistance))\n"
                    output += "Average distance: \(String(format: "%.4f", avgDistance))\n"
                    output += "Total frames: \(distances.count)\n"
                    
                    return output
                } else if let qwen25VL = context.model as? Qwen25VL {
                    return "Video similarity calculation not yet implemented for Qwen25VL"
                } else {
                    return "Error: Model is not Qwen2VL, Qwen3VL, or Qwen25VL"
                }
            }
            
            self.output = result
        } catch {
            self.output = "Failed to calculate video similarities: \(error)"
        }
    }
    
    func detectSceneChanges(videoURL: URL, threshold: Float, minSceneDuration: Float, maxSceneDuration: Float) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await detectSceneChangesAsync(videoURL: videoURL, threshold: threshold, minSceneDuration: minSceneDuration, maxSceneDuration: maxSceneDuration)
            running = false
        }
    }
    
    private func detectSceneChangesAsync(videoURL: URL, threshold: Float, minSceneDuration: Float, maxSceneDuration: Float) async {
        self.output = ""
        
        do {
            let modelContainer = try await load()
            
            let result = try await modelContainer.perform { (context: ModelContext) -> String in
                if let qwen2VL = context.model as? Qwen2VL {
                    guard let processorConfig = (context.processor as? Qwen2VLProcessor)?.config else {
                        return "Error: Processor is not Qwen2VLProcessor"
                    }
                
                let startTime = Date()
                let sceneChanges = try await qwen2VL.detectSceneChanges(from: videoURL, threshold: threshold, minSceneDuration: TimeInterval(minSceneDuration), maxSceneDuration: TimeInterval(maxSceneDuration), processorConfig: processorConfig)
                let endTime = Date()
                let duration = endTime.timeIntervalSince(startTime)
                
                var output = "Scene Change Detection Results:\n"
                output += "Threshold: \(String(format: "%.2f", threshold))\n"
                output += "Min scene duration: \(String(format: "%.1f", minSceneDuration))s\n"
                output += "Max scene duration: \(String(format: "%.1f", maxSceneDuration))s\n"
                output += "Total scenes detected: \(sceneChanges.count)\n"
                output += "Processing time: \(String(format: "%.2f", duration)) seconds\n\n"
                
                output += "Scene boundaries:\n"
                for (sceneIndex, (frameIndex, timestamp)) in sceneChanges.enumerated() {
                    output += "Scene \(sceneIndex + 1): starts at frame \(frameIndex) (\(String(format: "%.1f", timestamp))s)\n"
                }
                
                // Calculate scene durations if we have multiple scenes
                if sceneChanges.count > 1 {
                    output += "\nScene durations:\n"
                    for i in 0..<(sceneChanges.count - 1) {
                        let startFrame = sceneChanges[i].frameIndex
                        let startTime = sceneChanges[i].timestamp
                        let endFrame = sceneChanges[i + 1].frameIndex - 1
                        let endTime = sceneChanges[i + 1].timestamp - 0.5 // Subtract 0.5s since we're at 2 FPS
                        let frameDuration = endFrame - startFrame + 1
                        let timeDuration = endTime - startTime
                        output += "Scene \(i + 1): \(frameDuration) frames (\(String(format: "%.1f", timeDuration))s) - frames \(startFrame)-\(endFrame) (\(String(format: "%.1f", startTime))s-\(String(format: "%.1f", endTime))s)\n"
                    }
                    
                    // Last scene duration
                    let lastSceneStart = sceneChanges.last!
                    output += "Scene \(sceneChanges.count): from frame \(lastSceneStart.frameIndex) (\(String(format: "%.1f", lastSceneStart.timestamp))s) to end\n"
                }
                
                return output
                } else if let qwen3VL = context.model as? Qwen3VL {
                    guard let processorConfig = (context.processor as? Qwen3VLProcessor)?.config else {
                        return "Error: Processor is not Qwen3VLProcessor"
                    }
                
                let startTime = Date()
                let sceneChanges = try await qwen3VL.detectSceneChanges(from: videoURL, threshold: threshold, minSceneDuration: TimeInterval(minSceneDuration), maxSceneDuration: TimeInterval(maxSceneDuration), processorConfig: processorConfig)
                let endTime = Date()
                let duration = endTime.timeIntervalSince(startTime)
                
                var output = "Scene Change Detection Results:\n"
                output += "Threshold: \(String(format: "%.2f", threshold))\n"
                output += "Min scene duration: \(String(format: "%.1f", minSceneDuration))s\n"
                output += "Max scene duration: \(String(format: "%.1f", maxSceneDuration))s\n"
                output += "Total scenes detected: \(sceneChanges.count)\n"
                output += "Processing time: \(String(format: "%.2f", duration)) seconds\n\n"
                
                output += "Scene boundaries:\n"
                for (sceneIndex, (frameIndex, timestamp)) in sceneChanges.enumerated() {
                    output += "Scene \(sceneIndex + 1): starts at frame \(frameIndex) (\(String(format: "%.1f", timestamp))s)\n"
                }
                
                // Calculate scene durations if we have multiple scenes
                if sceneChanges.count > 1 {
                    output += "\nScene durations:\n"
                    for i in 0..<(sceneChanges.count - 1) {
                        let startFrame = sceneChanges[i].frameIndex
                        let startTime = sceneChanges[i].timestamp
                        let endFrame = sceneChanges[i + 1].frameIndex - 1
                        let endTime = sceneChanges[i + 1].timestamp - 0.5 // Subtract 0.5s since we're at 2 FPS
                        let frameDuration = endFrame - startFrame + 1
                        let timeDuration = endTime - startTime
                        output += "Scene \(i + 1): \(frameDuration) frames (\(String(format: "%.1f", timeDuration))s) - frames \(startFrame)-\(endFrame) (\(String(format: "%.1f", startTime))s-\(String(format: "%.1f", endTime))s)\n"
                    }
                    
                    // Last scene duration
                    let lastSceneStart = sceneChanges.last!
                    output += "Scene \(sceneChanges.count): from frame \(lastSceneStart.frameIndex) (\(String(format: "%.1f", lastSceneStart.timestamp))s) to end\n"
                }
                
                return output
                } else if let qwen25VL = context.model as? Qwen25VL {
                    return "Scene change detection not yet implemented for Qwen25VL"
                } else {
                    return "Error: Model is not Qwen2VL, Qwen3VL, or Qwen25VL"
                }
            }
            
            self.output = result
        } catch {
            self.output = "Failed to detect scene changes: \(error)"
        }
    }
    
    // MARK: - Vision Framework Methods
    
    func extractVisionFeaturePrints(videoURL: URL, frameSpecification: FrameSpecification) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await extractVisionFeaturePrintsAsync(videoURL: videoURL, frameSpecification: frameSpecification)
            running = false
        }
    }
    
    private func extractVisionFeaturePrintsAsync(videoURL: URL, frameSpecification: FrameSpecification) async {
        self.output = ""
        
        do {
            let startTime = Date()
            let featurePrints = try await visionProcessor.extractVideoFeaturePrints(from: videoURL, frameSpecification: frameSpecification)
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            
            var output = "Vision Feature Print Extraction Results:\n"
            output += "Frame specification: \(frameSpecification)\n"
            output += "Total feature prints extracted: \(featurePrints.count)\n"
            output += "Processing time: \(String(format: "%.2f", duration)) seconds\n\n"
            
            output += "Feature print details:\n"
            for (index, featurePrint) in featurePrints.enumerated() {
                output += "Frame \(index + 1): data length \(featurePrint.data.count) bytes\n"
            }
            
            self.output = output
        } catch {
            self.output = "Failed to extract Vision feature prints: \(error)"
        }
    }
    
    func calculateVisionDistances(videoURL: URL, frameSpecification: FrameSpecification) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await calculateVisionDistancesAsync(videoURL: videoURL, frameSpecification: frameSpecification)
            running = false
        }
    }
    
    private func calculateVisionDistancesAsync(videoURL: URL, frameSpecification: FrameSpecification) async {
        self.output = ""
        
        do {
            let startTime = Date()
            let distances = try await visionProcessor.calculateVideoFeaturePrintDistances(from: videoURL, frameSpecification: frameSpecification)
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            
            var output = "Vision Feature Print Distance Results:\n"
            output += "Frame specification: \(frameSpecification)\n"
            output += "Total distances calculated: \(distances.count)\n"
            output += "Processing time: \(String(format: "%.2f", duration)) seconds\n\n"
            
            output += "Distance values (using first frame as reference):\n"
            for (index, distance) in distances.enumerated() {
                output += "Frame \(index + 1): \(String(format: "%.4f", distance))\n"
            }
            
            // Calculate statistics
            if distances.count > 1 {
                let minDistance = distances.min() ?? 0.0
                let maxDistance = distances.max() ?? 0.0
                let avgDistance = distances.reduce(0, +) / Float(distances.count)
                
                output += "\nDistance statistics:\n"
                output += "Minimum distance: \(String(format: "%.4f", minDistance))\n"
                output += "Maximum distance: \(String(format: "%.4f", maxDistance))\n"
                output += "Average distance: \(String(format: "%.4f", avgDistance))\n"
            }
            
            self.output = output
        } catch {
            self.output = "Failed to calculate Vision distances: \(error)"
        }
    }
    
    func detectVisionSceneChanges(videoURL: URL, threshold: Float, minSceneDuration: Float, maxSceneDuration: Float) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await detectVisionSceneChangesAsync(videoURL: videoURL, threshold: threshold, minSceneDuration: minSceneDuration, maxSceneDuration: maxSceneDuration)
            running = false
        }
    }
    
    private func detectVisionSceneChangesAsync(videoURL: URL, threshold: Float, minSceneDuration: Float, maxSceneDuration: Float) async {
        self.output = ""
        
        do {
            let startTime = Date()
            let sceneChanges = try await visionProcessor.detectSceneChanges(from: videoURL, threshold: threshold, minSceneDuration: TimeInterval(minSceneDuration), maxSceneDuration: TimeInterval(maxSceneDuration))
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            
            var output = "Vision-Based Scene Change Detection Results:\n"
            output += "Threshold: \(String(format: "%.2f", threshold))\n"
            output += "Min scene duration: \(String(format: "%.1f", minSceneDuration))s\n"
            output += "Max scene duration: \(String(format: "%.1f", maxSceneDuration))s\n"
            output += "Total scenes detected: \(sceneChanges.count)\n"
            output += "Processing time: \(String(format: "%.2f", duration)) seconds\n\n"
            
            output += "Scene boundaries:\n"
            for (sceneIndex, (frameIndex, timestamp)) in sceneChanges.enumerated() {
                output += "Scene \(sceneIndex + 1): starts at frame \(frameIndex) (\(String(format: "%.1f", timestamp))s)\n"
            }
            
            // Calculate scene durations if we have multiple scenes
            if sceneChanges.count > 1 {
                output += "\nScene durations:\n"
                for i in 0..<(sceneChanges.count - 1) {
                    let startFrame = sceneChanges[i].frameIndex
                    let startTime = sceneChanges[i].timestamp
                    let endFrame = sceneChanges[i + 1].frameIndex - 1
                    let endTime = sceneChanges[i + 1].timestamp - 0.5 // Subtract 0.5s since we're at 2 FPS
                    let frameDuration = (endFrame - startFrame) + 1
                    let timeDuration = endTime - startTime
                    output += "Scene \(i + 1): \(frameDuration) frames (\(String(format: "%.1f", timeDuration))s) - frames \(startFrame)-\(endFrame) (\(String(format: "%.1f", startTime))s-\(String(format: "%.1f", endTime))s)\n"
                }
                
                // Last scene duration
                let lastSceneStart = sceneChanges.last!
                output += "Scene \(sceneChanges.count): from frame \(lastSceneStart.frameIndex) (\(String(format: "%.1f", lastSceneStart.timestamp))s) to end\n"
            }
            
            self.output = output
        } catch {
            self.output = "Failed to detect Vision-based scene changes: \(error)"
        }
    }
}

#if os(iOS) || os(visionOS)
    struct TransferableVideo: Transferable {
        let url: URL

        static var transferRepresentation: some TransferRepresentation {
            FileRepresentation(contentType: .movie) { movie in
                SentTransferredFile(movie.url)
            } importing: { received in
                let sandboxURL = try SandboxFileTransfer.transferFileToTemp(from: received.file)
                return .init(url: sandboxURL)
            }
        }
    }
#endif

struct SandboxFileTransfer {
    static func transferFileToTemp(from sourceURL: URL) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let sandboxURL = tempDir.appendingPathComponent(sourceURL.lastPathComponent)

        if FileManager.default.fileExists(atPath: sandboxURL.path()) {
            try FileManager.default.removeItem(at: sandboxURL)
        }

        try FileManager.default.copyItem(at: sourceURL, to: sandboxURL)
        return sandboxURL
    }
}

// MARK: - Vision Processor

/// A utility class for processing images and videos using Apple's Vision framework
/// 
/// This class provides functions for extracting feature prints from images and videos,
/// calculating distances between feature prints, and detecting scene changes in videos.
/// 
/// Example usage:
/// ```swift
/// let processor = VisionProcessor()
/// 
/// // Extract feature print from image
/// let image = CIImage(contentsOf: imageURL)!
/// let featurePrint = try processor.extractFeaturePrint(from: image)
/// 
/// // Calculate distance between two feature prints
/// let distance = try processor.featurePrintDistance(featurePrint1, featurePrint2)
/// 
/// // Detect scene changes in video
/// let sceneChanges = try await processor.detectSceneChanges(
///     from: videoURL, 
///     threshold: 0.5,
///     minSceneDuration: 2.0,
///     maxSceneDuration: 15.0
/// )
/// ```
public class VisionProcessor {
    
    public init() {}
    
    /// Extract Vision framework feature prints from a single image
    /// 
    /// This function uses Apple's Vision framework to generate feature prints
    /// which are high-dimensional vectors representing image content.
    /// 
    /// Example usage:
    /// ```swift
    /// let processor = VisionProcessor()
    /// let image = CIImage(contentsOf: imageURL)!
    /// let featurePrint = try processor.extractFeaturePrint(from: image)
    /// // featurePrint is a VNFeaturePrintObservation
    /// ```
    /// 
    /// - Parameter image: The input image as a CIImage
    /// - Returns: The feature print observation
    /// - Throws: Error if feature print generation fails
    public func extractFeaturePrint(from image: CIImage) throws -> VNFeaturePrintObservation {
        let request = VNGenerateImageFeaturePrintRequest()
        
        // Convert CIImage to CGImage for Vision framework
        let context = CIContext()
        guard let cgImage = context.createCGImage(image, from: image.extent) else {
            throw NSError(domain: "VisionProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to convert CIImage to CGImage"])
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
            
            guard let result = request.results?.first as? VNFeaturePrintObservation else {
                throw NSError(domain: "VisionProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to generate feature print"])
            }
            
            print("Vision feature print generated successfully")
            print("Feature print data length: \(result.data.count) bytes")
            
            return result
        } catch {
            throw NSError(domain: "VisionProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "Vision framework error: \(error.localizedDescription)"])
        }
    }

    /// Calculate distance between two Vision feature prints
    /// 
    /// This function computes the distance between two feature prints
    /// using Apple's Vision framework computeDistance method.
    /// 
    /// Example usage:
    /// ```swift
    /// let processor = VisionProcessor()
    /// let image1 = CIImage(contentsOf: imageURL1)!
    /// let image2 = CIImage(contentsOf: imageURL2)!
    /// 
    /// let featurePrint1 = try processor.extractFeaturePrint(from: image1)
    /// let featurePrint2 = try processor.extractFeaturePrint(from: image2)
    /// let distance = try processor.featurePrintDistance(featurePrint1, featurePrint2)
    /// // distance is a value where 0 means identical, higher values mean more different
    /// ```
    /// 
    /// - Parameter featurePrint1: First feature print observation
    /// - Parameter featurePrint2: Second feature print observation
    /// - Returns: Distance value (0 = identical, higher = more different)
    /// - Throws: Error if distance computation fails
    public func featurePrintDistance(_ featurePrint1: VNFeaturePrintObservation, _ featurePrint2: VNFeaturePrintObservation) throws -> Float {
        var distance: Float = 0.0
        
        do {
            try featurePrint1.computeDistance(&distance, to: featurePrint2)
            return distance
        } catch {
            throw NSError(domain: "VisionProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to compute feature print distance: \(error.localizedDescription)"])
        }
    }

    /// Extract Vision feature prints from a video by processing each frame
    /// 
    /// This function processes each frame of a video and extracts Vision framework
    /// feature prints for each frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let processor = VisionProcessor()
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// let frameFeaturePrints = try await processor.extractVideoFeaturePrints(from: videoURL)
    /// // frameFeaturePrints is an array of VNFeaturePrintObservation, one for each frame
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter frameSpecification: Which frames to process (frame numbers, timestamps, or all frames)
    /// - Returns: Array of feature print observations for each frame
    /// - Throws: Error if video processing fails
    public func extractVideoFeaturePrints(
        from videoURL: URL,
        frameSpecification: FrameSpecification = .allFrames
    ) async throws -> [VNFeaturePrintObservation] {
        // Get video asset and duration
        let asset = AVAsset(url: videoURL)
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)
        
        print("Video duration: \(String(format: "%.2f", durationSeconds)) seconds")
        
        // Determine which frames to process
        let framesToProcess: [Int]
        let frameTimestamps: [TimeInterval]
        let fps: Double = 2.0 // Default FPS for video processing
        
        switch frameSpecification {
        case .frameNumbers(let frameNumbers):
            framesToProcess = frameNumbers.sorted()
            frameTimestamps = frameNumbers.map { TimeInterval($0) / fps }
            print("Processing specific frame numbers: \(frameNumbers)")
            
        case .timestamps(let timestamps):
            let sortedTimestamps = timestamps.sorted()
            framesToProcess = sortedTimestamps.map { Int($0 * fps) }
            frameTimestamps = sortedTimestamps
            print("Processing frames at timestamps: \(timestamps.map { String(format: "%.2f", $0) })")
            
        case .allFrames:
            // Extract all frames as CIImage sequence
            let ciImages = try await MediaProcessing.asCIImageSequence(
                AVAsset(url: videoURL), 
                samplesPerSecond: Int(fps)
            )
            
            var frameFeaturePrints: [VNFeaturePrintObservation] = []
            
            // Process each frame
            for (index, frameImage) in ciImages.enumerated() {
                print("Processing frame \(index + 1)/\(ciImages.count)")
                
                let featurePrint = try extractFeaturePrint(from: frameImage)
                frameFeaturePrints.append(featurePrint)
            }
            
            print("Successfully extracted Vision feature prints for \(frameFeaturePrints.count) frames")
            return frameFeaturePrints
        }
        
        // Validate frame numbers
        let maxFrameNumber = Int(durationSeconds * fps)
        let validFrames = framesToProcess.filter { $0 >= 0 && $0 < maxFrameNumber }
        
        if validFrames.count != framesToProcess.count {
            let invalidFrames = framesToProcess.filter { $0 < 0 || $0 >= maxFrameNumber }
            print("Warning: Invalid frame numbers ignored: \(invalidFrames)")
        }
        
        guard !validFrames.isEmpty else {
            throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No valid frames to process"])
        }
        
        print("Processing \(validFrames.count) valid frames out of \(framesToProcess.count) requested")
        
        // Extract specific frames and generate feature prints
        var frameFeaturePrints: [VNFeaturePrintObservation] = []
        
        for (index, frameNumber) in validFrames.enumerated() {
            let timestamp = frameTimestamps[index]
            print("Processing frame \(frameNumber) at timestamp \(String(format: "%.2f", timestamp))s (\(index + 1)/\(validFrames.count))")
            
            // Extract single frame at specific timestamp
            let frameImage = try await extractFrameFromAsset(asset, at: timestamp)
            
            let featurePrint = try extractFeaturePrint(from: frameImage)
            frameFeaturePrints.append(featurePrint)
        }
        
        print("Successfully extracted Vision feature prints for \(frameFeaturePrints.count) specified frames")
        return frameFeaturePrints
    }

    /// Calculate Vision feature print distances between specified frames and the first frame as reference
    /// 
    /// This function extracts Vision feature prints from specified frames of a video
    /// and calculates distances between each frame and the first frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let processor = VisionProcessor()
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// 
    /// // Calculate distances for specific frames
    /// let distances = try await processor.calculateVideoFeaturePrintDistances(
    ///     from: videoURL, 
    ///     frameSpecification: .frameNumbers([0, 10, 20, 30])
    /// )
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter frameSpecification: Which frames to process (frame numbers, timestamps, or all frames)
    /// - Returns: Array of distance values for each frame (first frame will be 0.0)
    /// - Throws: Error if video processing fails
    public func calculateVideoFeaturePrintDistances(
        from videoURL: URL,
        frameSpecification: FrameSpecification = .allFrames
    ) async throws -> [Float] {
        // Extract Vision feature prints from specified frames
        let frameFeaturePrints = try await extractVideoFeaturePrints(
            from: videoURL,
            frameSpecification: frameSpecification
        )
        
        guard !frameFeaturePrints.isEmpty else {
            throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No frames extracted from video"])
        }
        
        let referenceFeaturePrint = frameFeaturePrints[0]
        var distances: [Float] = []
        
        // Calculate distance between each frame and the reference frame
        for (index, featurePrint) in frameFeaturePrints.enumerated() {
            let distance = try featurePrintDistance(referenceFeaturePrint, featurePrint)
            distances.append(distance)
            
            print("Frame \(index + 1) Vision distance to reference: \(distance)")
        }
        
        print("Successfully calculated Vision feature print distances for \(distances.count) frames")
        return distances
    }

    /// Detect scene changes in a video using Vision framework feature prints
    /// 
    /// This function analyzes each frame of a video and detects scene changes
    /// by comparing Vision feature prints between frames.
    /// 
    /// Example usage:
    /// ```swift
    /// let processor = VisionProcessor()
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// let sceneChanges = try await processor.detectSceneChanges(
    ///     from: videoURL, 
    ///     threshold: 0.5, 
    ///     minSceneDuration: 2.0,
    ///     maxSceneDuration: 15.0
    /// )
    /// // sceneChanges contains frame indices where scene changes occur
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter threshold: Feature print distance threshold for scene change detection (default: 0.5)
    /// - Parameter minSceneDuration: Minimum scene duration in seconds (default: 2.0)
    /// - Parameter maxSceneDuration: Maximum scene duration in seconds (default: 15.0)
    /// - Returns: Array of frame indices where scene changes occur (including frame 0)
    /// - Throws: Error if video processing fails
    public func detectSceneChanges(
        from videoURL: URL,
        threshold: Float = 0.5,
        minSceneDuration: TimeInterval = 2.0,
        maxSceneDuration: TimeInterval = 15.0
    ) async throws -> [(frameIndex: Int, timestamp: TimeInterval)] {
        let startTime = Date()
        
        // Extract Vision feature prints from video at 2 FPS for scene detection
        let frameFeaturePrints = try await extractVideoFeaturePrints(
            from: videoURL,
            frameSpecification: .allFrames
        )
        
        guard !frameFeaturePrints.isEmpty else {
            throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No frames extracted from video"])
        }
        
        var sceneChanges: [(frameIndex: Int, timestamp: TimeInterval)] = [(0, 0.0)] // Always include frame 0 as first scene
        var currentReferenceFeaturePrint = frameFeaturePrints[0]
        var allDistances: [Float] = []
        
        print("Vision-based scene change detection with threshold: \(threshold), min scene duration: \(minSceneDuration)s, max scene duration: \(maxSceneDuration)s")
        print("DEBUG: minSceneDuration = \(minSceneDuration), maxSceneDuration = \(maxSceneDuration)")
        print("Frame 0 (0.0s): Starting new scene (reference frame)")
        
        // Analyze each frame starting from frame 1
        for frameIndex in 1..<frameFeaturePrints.count {
            let currentFeaturePrint = frameFeaturePrints[frameIndex]
            let timestamp = TimeInterval(frameIndex) * 0.5 // 2 FPS = 0.5 seconds per frame
            let timeSinceLastScene = timestamp - sceneChanges.last!.timestamp
            
            let distance = try featurePrintDistance(currentReferenceFeaturePrint, currentFeaturePrint)
            allDistances.append(distance)
            
            print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): Vision distance to reference = \(String(format: "%.6f", distance)), time since last scene: \(String(format: "%.1f", timeSinceLastScene))s")
            
            var sceneChangeDetected = false
            var sceneChangeReason = ""
            
            // Check if maximum scene duration has been exceeded
            if timeSinceLastScene >= maxSceneDuration {
                sceneChangeDetected = true
                sceneChangeReason = "max duration exceeded"
                print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): FORCED SCENE CHANGE - Max duration exceeded (\(String(format: "%.1f", timeSinceLastScene))s >= \(maxSceneDuration)s)")
            }
            // Check if distance threshold is exceeded and minimum duration is met
            else if distance > threshold && timeSinceLastScene >= minSceneDuration {
                sceneChangeDetected = true
                sceneChangeReason = "distance threshold"
                print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): SCENE CHANGE DETECTED - Distance threshold exceeded (duration: \(String(format: "%.1f", timeSinceLastScene))s)")
            }
            // Check if distance threshold is exceeded but minimum duration is not met
            else if distance > threshold && timeSinceLastScene < minSceneDuration {
                print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): Scene change ignored - too short (duration: \(String(format: "%.1f", timeSinceLastScene))s < \(minSceneDuration)s)")
            }
            
            if sceneChangeDetected {
                sceneChanges.append((frameIndex: frameIndex, timestamp: timestamp))
                currentReferenceFeaturePrint = currentFeaturePrint
                print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): SCENE CHANGE APPLIED - \(sceneChangeReason) (duration: \(String(format: "%.1f", timeSinceLastScene))s)")
            }
        }
        
        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        
        print("Vision-based scene change detection complete!")
        print("Total scenes detected: \(sceneChanges.count)")
        print("Scene changes at frames and timestamps:")
        for (frameIndex, timestamp) in sceneChanges {
            print("  Frame \(frameIndex): \(String(format: "%.1f", timestamp))s")
        }
        print("Total processing time: \(String(format: "%.2f", duration)) seconds")
        print("Average time per frame: \(String(format: "%.3f", duration / Double(frameFeaturePrints.count))) seconds")
        
        return sceneChanges
    }
}

// MARK: - Frame Specification

/// Specification for which frames to process in video operations
public enum FrameSpecification {
    case allFrames
    case frameNumbers([Int])
    case timestamps([TimeInterval])
}

// MARK: - Helper Functions

/// Extract a single frame from an AVAsset at a specific timestamp
private func extractFrameFromAsset(_ asset: AVAsset, at timestamp: TimeInterval) async throws -> CIImage {
    let imageGenerator = AVAssetImageGenerator(asset: asset)
    imageGenerator.appliesPreferredTrackTransform = true
    imageGenerator.requestedTimeToleranceBefore = .zero
    imageGenerator.requestedTimeToleranceAfter = .zero
    
    let time = CMTime(seconds: timestamp, preferredTimescale: 600)
    
    do {
        let cgImage = try await imageGenerator.image(at: time).image
        let ciImage = CIImage(cgImage: cgImage)
        return ciImage
    } catch {
        throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to extract frame at timestamp \(timestamp): \(error.localizedDescription)"])
    }
}


