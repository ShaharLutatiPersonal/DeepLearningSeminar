/**********************************************************************************************************************/
/* Description:                                                                                                       */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* Own header */
#include "bsp_objects.h"

/* C Standards */

/* Project dependencies */
#include <gpio.h>
#include <i2s_interrupt.h>
#include <uart.h>

/* External dependencies */

/**********************************************************************************************************************/
/* Objects                                                                                                            */
/**********************************************************************************************************************/
CREATE_GPIO_PIN(LED_PIN, LED1_PORT, LED1_PIN);
CREATE_GPIO_PIN(BUTTON0_PIN, BUTTON0_PORT, BUTTON0_PIN);
CREATE_GPIO_PIN(BUTTON1_PIN, BUTTON1_PORT, BUTTON1_PIN);
CREATE_GPIO_PIN(MIC_ENABLE, GPIO_MIC_ENABLE_PORT, GPIO_MIC_ENABLE_PIN);
CREATE_GPIO_DRIVER;

CREATE_GPIO_PIN(UART0_RX, GPIO_UART0_RX_PORT, GPIO_UART0_RX_PIN);
CREATE_GPIO_PIN(UART0_TX, GPIO_UART0_TX_PORT, GPIO_UART0_TX_PIN);
ARM_USART_CONF uart0_conf = {
    .baud_rate = 115200,
};

CREATE_UART_DRIVER(0);

CREATE_GPIO_PIN(I2S1_RX, GPIO_I2S1_RX_PORT, GPIO_I2S1_RX_PIN);
CREATE_GPIO_PIN(I2S1_CLK, GPIO_I2S1_CLK_PORT, GPIO_I2S1_CLK_PIN);
CREATE_GPIO_PIN(I2S1_WS, GPIO_I2S1_WS_PORT, GPIO_I2S1_WS_PIN);
#define GPIO_I2S1_TX (*(ARM_GPIO_PIN *)ARM_GPIO_NOT_CONNECTED)

ARM_SAI_CONF i2s1_conf = {
    .baudrate          = MICROPHONE_SAMPLE_FREQ * 64,
    .databits          = 24,
    .word_size         = 32,
    .is_mono           = false,
    .is_left_justified = true,
    .is_master         = true,
};

CREATE_I2S_INTERRUPT_DRIVER(1);
