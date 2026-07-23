# speech2textconverter
Uses whisper.cpp and ffmpeg to convert any audio input to json text output.

Requires ffmpeg to be installed on the system and a clone of whisper.cpp should be built and placed in directory.
Run make to build the project.

Execute with the following command in the terminal:
./transcribe_pipeline input_audio_path output_json_path model_path
