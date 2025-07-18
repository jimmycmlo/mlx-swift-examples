// Copyright 2024 Apple Inc.

import AVKit
import AsyncAlgorithms
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import PhotosUI
import SwiftUI

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
    @State private var sceneThreshold: Float = 0.05

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
            }
            
            if selectedVideoURL != nil {
                VStack(spacing: 16) {
                    Text("Scene Change Detection Settings")
                        .font(.headline)
                        .padding(.top)
                    
                    VStack(alignment: .center, spacing: 12) {
                        Text("Threshold: \(String(format: "%.2f", sceneThreshold))")
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
                    llm.generate(image: nil, videoURL: videoURL)
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
                llm.extractVideoPatches(videoURL: videoURL)
            }
        }
    }
    
    private func videoMeanPool() {
        Task {
            if let videoURL = selectedVideoURL {
                llm.videoMeanPool(videoURL: videoURL)
            }
        }
    }
    
    private func calculateVideoSimilarity() {
        Task {
            if let videoURL = selectedVideoURL {
                llm.calculateVideoSimilarity(videoURL: videoURL)
            }
        }
    }
    
    private func detectSceneChanges() {
        Task {
            if let videoURL = selectedVideoURL {
                llm.detectSceneChanges(videoURL: videoURL, threshold: sceneThreshold)
            }
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

    /// This controls which model loads. `smolvlm` is very small even unquantized, so it will fit on
    /// more devices.
//    let modelConfiguration = VLMRegistry.smolvlm
    let modelConfiguration = VLMRegistry.qwen2VL2BInstruct4Bit

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

    private func generate(prompt: String, image: CIImage?, videoURL: URL?) async {

        self.output = ""

        do {
            let modelContainer = try await load()

            // Set pixel limits based on media type
            await modelContainer.update { context in
                if let qwen2vlProcessor = context.processor as? Qwen2VLProcessor {
                    if videoURL != nil {
                        // For videos: lower pixel limits
                        qwen2vlProcessor.config.maxPixels = 256 * 28 * 28  // 200,704
                        qwen2vlProcessor.config.minPixels = 64 * 28 * 28   // 50,176
                    } else if image != nil {
                        // For images: higher pixel limits
                        qwen2vlProcessor.config.maxPixels = 1024 * 28 * 28  // 401,408
                        qwen2vlProcessor.config.minPixels = 256 * 28 * 28  // 200,704
                    }
                }
            }

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

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

                let lmInput = try await context.processor.prepare(input: userInput)

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

    func generate(image: CIImage?, videoURL: URL?) {
        guard !running else { return }
        let currentPrompt = prompt
        prompt = ""
        generationTask = Task {
            running = true
            await generate(prompt: currentPrompt, image: image, videoURL: videoURL)
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
    
    func extractVideoPatches(videoURL: URL) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await extractVideoPatchesAsync(videoURL: videoURL)
            running = false
        }
    }
    
    private func extractVideoPatchesAsync(videoURL: URL) async {
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
                
                let frameEmbeddings = try await qwen2VL.extractVideoPatchEmbeddings(from: videoURL, processorConfig: processorConfig)
                
                var output = "Successfully extracted patch embeddings for \(frameEmbeddings.count) frames:\n"
                for (index, embedding) in frameEmbeddings.enumerated() {
                    output += "Frame \(index + 1): shape \(embedding.shape), size \(embedding.size)\n"
                }
                
                return output
            }
            
            self.output = result
        } catch {
            self.output = "Failed to extract video patches: \(error)"
        }
    }
    
    func videoMeanPool(videoURL: URL) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await videoMeanPoolAsync(videoURL: videoURL)
            running = false
        }
    }
    
    private func videoMeanPoolAsync(videoURL: URL) async {
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
                
                let frameEmbeddings = try await qwen2VL.extractAndPoolVideoEmbeddings(from: videoURL, processorConfig: processorConfig)
                
                var output = "Successfully extracted and mean-pooled embeddings for \(frameEmbeddings.count) frames:\n"
                for (index, embedding) in frameEmbeddings.enumerated() {
                    output += "Frame \(index + 1): shape \(embedding.shape), size \(embedding.size)\n"
                }
                
                return output
            }
            
            self.output = result
        } catch {
            self.output = "Failed to video mean pool: \(error)"
        }
    }
    
    func calculateVideoSimilarity(videoURL: URL) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await calculateVideoSimilarityAsync(videoURL: videoURL)
            running = false
        }
    }
    
    private func calculateVideoSimilarityAsync(videoURL: URL) async {
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
                
                let distances = try await qwen2VL.calculateVideoFrameDistances(from: videoURL, processorConfig: processorConfig)
                
                var output = "Cosine distances to reference frame (Frame 1):\n"
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
            }
            
            self.output = result
        } catch {
            self.output = "Failed to calculate video similarities: \(error)"
        }
    }
    
    func detectSceneChanges(videoURL: URL, threshold: Float) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await detectSceneChangesAsync(videoURL: videoURL, threshold: threshold)
            running = false
        }
    }
    
    private func detectSceneChangesAsync(videoURL: URL, threshold: Float) async {
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
                
                let startTime = Date()
                let sceneChanges = try await qwen2VL.detectSceneChanges(from: videoURL, threshold: threshold, processorConfig: processorConfig)
                let endTime = Date()
                let duration = endTime.timeIntervalSince(startTime)
                
                var output = "Scene Change Detection Results:\n"
                output += "Threshold: \(String(format: "%.2f", threshold))\n"
                output += "Total scenes detected: \(sceneChanges.count)\n"
                output += "Processing time: \(String(format: "%.2f", duration)) seconds\n\n"
                
                output += "Scene boundaries:\n"
                for (sceneIndex, frameIndex) in sceneChanges.enumerated() {
                    output += "Scene \(sceneIndex + 1): starts at frame \(frameIndex)\n"
                }
                
                // Calculate scene durations if we have multiple scenes
                if sceneChanges.count > 1 {
                    output += "\nScene durations (in frames):\n"
                    for i in 0..<(sceneChanges.count - 1) {
                        let startFrame = sceneChanges[i]
                        let endFrame = sceneChanges[i + 1] - 1
                        let duration = endFrame - startFrame + 1
                        output += "Scene \(i + 1): \(duration) frames (frames \(startFrame)-\(endFrame))\n"
                    }
                    
                    // Last scene duration
                    let lastSceneStart = sceneChanges.last!
                    output += "Scene \(sceneChanges.count): from frame \(lastSceneStart) to end\n"
                }
                
                return output
            }
            
            self.output = result
        } catch {
            self.output = "Failed to detect scene changes: \(error)"
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
