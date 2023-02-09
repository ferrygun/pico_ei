/*
 * Copyright (c) 2021 Arm Limited and Contributors. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 * 
 * 
 * This examples captures data from an analog microphone using a sample
 * rate of 8 kHz and prints the sample values over the USB serial
 * connection.
 */
#include <stdio.h>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "pico/stdlib.h"
#include "analog_microphone.h"
#include "analog_microphone.c"
#include "tusb.h"

// configuration
const struct analog_microphone_config config = {
    // GPIO to use for input, must be ADC compatible (GPIO 26 - 28)
    .gpio = 26,

    // bias voltage of microphone in volts
    .bias_voltage = 1.25,

    // sample rate in Hz
    .sample_rate = 8000,

    // number of samples to buffer
    .sample_buffer_size = 256,
};

// variables
#define INSIZE 256
int16_t sample_buffer[INSIZE];
volatile int samples_read = 0;

float features[INSIZE];

void on_analog_samples_ready() {
    // callback from library when all the samples in the library
    // internal sample buffer are ready for reading 
    samples_read = analog_microphone_read(sample_buffer, 256);
}

int raw_feature_get_data(size_t offset, size_t length, float * out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}


int main(void) {
    // initialize stdio and wait for USB CDC connect
    stdio_init_all();
    stdio_usb_init();

    ei_impulse_result_t result = {
        nullptr
    };

    while (!tud_cdc_connected()) {
        tight_loop_contents();
    }

    ei_printf("Edge Impulse standalone inferencing (Raspberry Pi Pico)\n");

    // initialize the analog microphone
    if (analog_microphone_init( & config) < 0) {
        ei_printf("analog microphone initialization failed!\n");
        while (1) {
            tight_loop_contents();
        }
    }

    // set callback that is called when all the samples in the library
    // internal sample buffer are ready for reading
    analog_microphone_set_samples_ready_handler(on_analog_samples_ready);

    // start capturing data from the analog microphone
    if (analog_microphone_start() < 0) {
        ei_printf("PDM microphone start failed!\n");
        while (1) {
            tight_loop_contents();
        }
    }

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE)
    {
      ei_printf("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
      return 1;
    }
    
    while (1) {
        // wait for new samples
        while (samples_read == 0) {
            tight_loop_contents();
        }

        // store and clear the samples read from the callback
        int sample_count = samples_read;
        samples_read = 0;
        
        // loop through any new collected samples
        for (int i = 0; i < sample_count; i++) {
            features[i] = sample_buffer[i];
        }

        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;


        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);

        ei_printf("run_classifier returned: %d\n", res);

        if (res != 0)
            return 1;

        ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);

        // print the predictions
        ei_printf("[");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("%.5f", result.classification[ix].value);
            #if EI_CLASSIFIER_HAS_ANOMALY == 1
            ei_printf(", ");
            #else
            if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
                ei_printf(", ");
            }
            #endif
        }
        #if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf("%.3f", result.anomaly);
        #endif
        printf("]\n");

        ei_sleep(2000);

    }

    return 0;
}
