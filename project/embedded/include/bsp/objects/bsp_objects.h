#pragma once
/**********************************************************************************************************************/
/* Description:                                                                                                       */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* C Standards */

/* Project dependencies */
#include <Driver_GPIO.h>
#include <Driver_SAI.h>
#include <Driver_USART.h>

/* External dependencies */

/**********************************************************************************************************************/
/* Macros                                                                                                             */
/**********************************************************************************************************************/
#define LED1_PORT (gpioPortD)
#define LED1_PIN  (8)

#define BUTTON0_PORT (gpioPortD)
#define BUTTON0_PIN  (14)

#define BUTTON1_PORT (gpioPortD)
#define BUTTON1_PIN  (15)

#define GPIO_UART0_TX_PORT (gpioPortA)
#define GPIO_UART0_TX_PIN  (0)

#define GPIO_UART0_RX_PORT (gpioPortA)
#define GPIO_UART0_RX_PIN  (1)

#define UART0_RX_LOC (USART_ROUTELOC0_RXLOC_LOC0)
#define UART0_TX_LOC (USART_ROUTELOC0_TXLOC_LOC0)

#define GPIO_I2S1_RX_PORT (gpioPortC)
#define GPIO_I2S1_RX_PIN  (7)

#define GPIO_I2S1_CLK_PORT (gpioPortC)
#define GPIO_I2S1_CLK_PIN  (8)

#define GPIO_I2S1_WS_PORT (gpioPortC)
#define GPIO_I2S1_WS_PIN  (9)

#define I2S1_RX_LOC            (USART_ROUTELOC0_RXLOC_LOC11)
#define I2S1_CLK_LOC           (USART_ROUTELOC0_CLKLOC_LOC11)
#define I2S1_WS_LOC            (USART_ROUTELOC0_CSLOC_LOC11)
#define I2S1_TX_LOC            (USART_ROUTELOC0_TXLOC_LOC11)
#define MICROPHONE_SAMPLE_FREQ (8000)

#define GPIO_MIC_ENABLE_PORT (gpioPortF)
#define GPIO_MIC_ENABLE_PIN  (10)

/**********************************************************************************************************************/
/* Objects                                                                                                            */
/**********************************************************************************************************************/
extern ARM_GPIO_PIN     GPIO_LED_PIN;
extern ARM_GPIO_PIN     GPIO_BUTTON0_PIN;
extern ARM_GPIO_PIN     GPIO_BUTTON1_PIN;
extern ARM_GPIO_PIN     GPIO_MIC_ENABLE;
extern ARM_GPIO_PIN     GPIO_UART0_RX;
extern ARM_GPIO_PIN     GPIO_UART0_TX;
extern ARM_DRIVER_USART Driver_UART_0;
extern ARM_DRIVER_SAI   Driver_SAI_1;