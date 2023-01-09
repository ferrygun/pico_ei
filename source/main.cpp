/* Includes ---------------------------------------------------------------- */
#include <stdio.h>
#include "pico/stdlib.h"
#include "ei_run_classifier.h"
#include "hardware/gpio.h"
#include "hardware/adc.h"
#include "pico/cyw43_arch.h"

#define FREQUENCY_HZ        50
#define INTERVAL_MS         (1000 / (FREQUENCY_HZ + 1))

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f
#define G0 1.65f
#define NSAMP 10
 
char ssid[] = "orangejuzz";
char pass[] = "Beetroot18";

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
 
const float conversion_factor = 3.3f / (1 << 12);
 
 
float readAxisAccelation (int adc_n) {
    adc_select_input(adc_n);
    unsigned int axis_raw = 0;
    for (int i=0;i<NSAMP;i++){
        axis_raw = axis_raw + adc_read();
        sleep_ms(1);
    }
    axis_raw = axis_raw/NSAMP;
    float axis_g = (axis_raw*conversion_factor)-G0;
    return axis_g;
}
 
int main()
{
    stdio_init_all();
    
    if (cyw43_arch_init_with_country(CYW43_COUNTRY_UK)) {
        printf("failed to initialise\n");
        return 1;
    }

    printf("initialised\n");

    cyw43_arch_enable_sta_mode();
    if (cyw43_arch_wifi_connect_timeout_ms(ssid, pass, CYW43_AUTH_WPA2_AES_PSK, 10000)) {
        printf("failed to connect\n");
        return 1;
    }

    printf("connected\n");


    adc_init();
    adc_gpio_init(28);
    
    ei_printf("EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE: %.3f\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME: %.3f\n", EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);

    while (true){
         
        ei_printf("\nStarting inferencing in 2 seconds...\n");
        sleep_ms(2000);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        ei_printf("Sampling...\n");

        // Allocate a buffer here for the values we'll read from the IMU
        float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

        for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
            // Determine the next tick (and then sleep later)

            uint64_t next_tick = ei_read_timer_us() + (EI_CLASSIFIER_INTERVAL_MS * 1000);
            //ei_printf("Loop: %.3f\n", next_tick - ei_read_timer_us());

            buffer[ix] = readAxisAccelation (2);
            buffer[ix + 1] = readAxisAccelation (2);
            buffer[ix + 2] = readAxisAccelation (2);
 
            buffer[ix + 0] *= CONVERT_G_TO_MS2;
            buffer[ix + 1] *= CONVERT_G_TO_MS2 * 0.2;
            buffer[ix + 2] *= CONVERT_G_TO_MS2 * 0.5;

 
            //sleep_us(next_tick - ei_read_timer_us());
        }
 
        // Turn the raw buffer in a signal which we can the classify
        signal_t signal;
        int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
        if (err != 0) {
            ei_printf("Failed to create signal from buffer (%d)\n", err);
            return 1;
        }

        // Run the classifier
        ei_impulse_result_t result = { 0 };
 
        err = run_classifier(&signal, &result, debug_nn);
        if (err != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", err);
            return 1;
        }
 
        // print the predictions
        ei_printf("Predictions ");

        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        ei_printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
        }

        #if EI_CLASSIFIER_HAS_ANOMALY == 1
            ei_printf("    anomaly score: %.3f\n", result.anomaly);
        #endif
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);

        //sleep_ms(INTERVAL_MS);
    }


return 0;
}
