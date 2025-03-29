def convert_model_to_header(model_file, output_header_file):
    with open(model_file, 'rb') as f:
        model_data = f.read()

    with open(output_header_file, 'w') as out_f:
        out_f.write('#ifndef _MODEL_DATA_H_\n')
        out_f.write('#define _MODEL_DATA_H_\n\n')
        out_f.write('const unsigned char model_data[] = {\n')
        
        for i in range(0, len(model_data), 16):
            line = model_data[i:i + 16]
            hex_values = ', '.join(f'0x{byte:02X}' for byte in line)
            out_f.write(f'  {hex_values},\n')

        out_f.write('};\n\n')
        out_f.write('#endif  // _MODEL_DATA_H_\n')

# Run this script on your model
convert_model_to_header('85_small.tflite', '85_small_tflite.h')
