#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <filesystem>

#include "whisper.cpp/include/whisper.h"
#include <json.hpp>

using json = nlohmann::json;

// Helper: Convert Whisper 10ms timestamp units to seconds, scaled by global time offset
double to_seconds(int64_t t, double offset_seconds) {
    return offset_seconds + (static_cast<double>(t) / 100.0);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./transcribe_pipeline <input_audio_path> <output_json_path> [model_path]\n";
        return 1;
    }

    std::string input_audio = argv[1];
    std::string output_json_path = argv[2];
    std::string model_path = (argc > 3) ? argv[3] : "whisper.cpp/models/ggml-base.en.bin";

    // Step A: Initialize Whisper Context
    std::cout << "[Pipeline] Initializing Whisper model from " << model_path << "...\n";
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);

    if (!ctx) {
        std::cerr << "[Error] Failed to initialize Whisper context.\n";
        return 1;
    }

    // Step B: Set up FFmpeg stream pipe
    // Output format: raw s16le (16-bit PCM), 1 channel (mono), 16000Hz sample rate
    std::string ffmpeg_cmd = "ffmpeg -loglevel error -i \"" + input_audio + 
                             "\" -f s16le -ac 1 -ar 16000 pipe:1";

    std::cout << "[Pipeline] Opening FFmpeg audio stream pipe...\n";
    FILE* pipe = popen(ffmpeg_cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "[Error] Failed to execute FFmpeg command pipe.\n";
        whisper_free(ctx);
        return 1;
    }

    // Step C: Set up Whisper inference parameters
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.n_threads        = 4;
    wparams.token_timestamps = true; // Enable word timestamps

    // Step D: Stream Ingestion and Chunked Processing Setup
    // 30 seconds of 16kHz audio = 480,000 samples
    const size_t SAMPLES_PER_CHUNK = 16000 * 30;
    std::vector<int16_t> pcm16_chunk(SAMPLES_PER_CHUNK);
    std::vector<float> pcmf32_chunk(SAMPLES_PER_CHUNK);

    json output_json;
    output_json["file"] = input_audio;
    output_json["words"] = json::array();

    std::string full_transcript = "";
    double current_time_offset = 0.0;
    size_t chunk_index = 0;

    std::cout << "[Pipeline] Streaming and transcribing audio in 30-second windows...\n";

    // Read audio in 30-second PCM frame blocks
    while (true) {
        size_t samples_read = fread(pcm16_chunk.data(), sizeof(int16_t), SAMPLES_PER_CHUNK, pipe);

        if (samples_read == 0) {
            break; // End of audio stream reached
        }

        // Convert int16_t PCM [-32768, 32767] to float [-1.0, 1.0] expected by Whisper
        pcmf32_chunk.resize(samples_read);
        for (size_t i = 0; i < samples_read; ++i) {
            pcmf32_chunk[i] = static_cast<float>(pcm16_chunk[i]) / 32768.0f;
        }

        // Run inference on the current 30s chunk
        if (whisper_full(ctx, wparams, pcmf32_chunk.data(), pcmf32_chunk.size()) != 0) {
            std::cerr << "[Warning] Failed to transcribe chunk " << chunk_index << "\n";
            continue;
        }

        // Extract tokens from current chunk and adjust global word timestamps
        int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            int n_tokens = whisper_full_n_tokens(ctx, i);

            for (int j = 0; j < n_tokens; ++j) {
                whisper_token_data token_data = whisper_full_get_token_data(ctx, i, j);
                const char* token_text = whisper_full_get_token_text(ctx, i, j);

                // Skip special control tokens
                if (token_data.id >= whisper_token_eot(ctx)) {
                    continue;
                }

                std::string word_str(token_text);
                full_transcript += word_str;

                json word_entry;
                word_entry["word"] = word_str;
                word_entry["start"] = to_seconds(token_data.t0, current_time_offset);
                word_entry["end"] = to_seconds(token_data.t1, current_time_offset);
                word_entry["probability"] = token_data.p;

                output_json["words"].push_back(word_entry);
            }
        }

        // Increment time offset for next chunk (samples_read / sample_rate)
        current_time_offset += static_cast<double>(samples_read) / 16000.0;
        chunk_index++;
    }

    pclose(pipe); // Close FFmpeg stream

    output_json["transcript"] = full_transcript;

    // Step E: Save Final Output JSON
    std::ofstream out_file(output_json_path);
    if (!out_file.is_open()) {
        std::cerr << "[Error] Could not open output file: " << output_json_path << "\n";
        whisper_free(ctx);
        return 1;
    }

    out_file << output_json.dump(4);
    out_file.close();

    std::cout << "[Success] Transcribed " << chunk_index << " chunk(s) (~" 
              << static_cast<int>(current_time_offset) << "s total).\n";
    std::cout << "[Success] JSON saved to: " << output_json_path << "\n";

    whisper_free(ctx);
    return 0;
}