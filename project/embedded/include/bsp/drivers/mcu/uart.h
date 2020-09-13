#pragma once
/**********************************************************************************************************************/
/* Description: Silabs EFR32 UART driver header */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* C Standards */
#include <stdint.h>

/* Project dependencies */
#include <Driver_USART.h>
#include <em_cmu.h>
#include <em_usart.h>
#include <gpio.h>

/**********************************************************************************************************************/
/* Macros                                                                                                             */
/**********************************************************************************************************************/
#define CREATE_UART_DRIVER(idx)                                                                                        \
    extern ARM_DRIVER_USART Driver_UART_##idx;                                                                         \
    extern uart_resources_t uart_##idx##_resources;                                                                    \
                                                                                                                       \
    static int32_t _uart_##idx##_driver_initialize(ARM_USART_SignalEvent_t cb_event)                                   \
    {                                                                                                                  \
        return uart_driver_initialize(&uart_##idx##_resources, cb_event);                                              \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _uart_##idx##_driver_uninitialize(void)                                                             \
    {                                                                                                                  \
        return uart_driver_uninitialize(&uart_##idx##_resources);                                                      \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _uart_##idx##_driver_power_control(ARM_POWER_STATE state)                                           \
    {                                                                                                                  \
        return uart_driver_power_control(&uart_##idx##_resources, state);                                              \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _uart_##idx##_driver_send(const void *data, uint32_t num)                                           \
    {                                                                                                                  \
        return uart_driver_send(&uart_##idx##_resources, data, num);                                                   \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _uart_##idx##_driver_receive(void *data, uint32_t num)                                              \
    {                                                                                                                  \
        return uart_driver_receive(&uart_##idx##_resources, data, num);                                                \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _uart_##idx##_driver_transfer(const void *data_out, void *data_in, uint32_t num)                    \
    {                                                                                                                  \
        return uart_driver_transfer(&uart_##idx##_resources, data_out, data_in, num);                                  \
    }                                                                                                                  \
                                                                                                                       \
    static uint32_t _uart_##idx##_driver_get_tx_count(void)                                                            \
    {                                                                                                                  \
        return uart_driver_get_tx_count(&uart_##idx##_resources);                                                      \
    }                                                                                                                  \
                                                                                                                       \
    static uint32_t _uart_##idx##_driver_get_rx_count(void)                                                            \
    {                                                                                                                  \
        return uart_driver_get_rx_count(&uart_##idx##_resources);                                                      \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _uart_##idx##_driver_control(uint32_t control, uint32_t arg)                                        \
    {                                                                                                                  \
        return uart_driver_control(&uart_##idx##_resources, control, arg);                                             \
    }                                                                                                                  \
                                                                                                                       \
    static ARM_USART_STATUS _uart_##idx##_driver_get_status(void)                                                      \
    {                                                                                                                  \
        return uart_driver_get_status(&uart_##idx##_resources);                                                        \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _uart_##idx##_driver_set_modem_control(ARM_USART_MODEM_CONTROL control)                             \
    {                                                                                                                  \
        return uart_driver_set_modem_control(&uart_##idx##_resources, control);                                        \
    }                                                                                                                  \
                                                                                                                       \
    static ARM_USART_MODEM_STATUS _uart_##idx##_driver_get_modem_status(void)                                          \
    {                                                                                                                  \
        return uart_driver_get_modem_status(&uart_##idx##_resources);                                                  \
    }                                                                                                                  \
                                                                                                                       \
    void USART##idx##_RX_IRQHandler(void)                                                                              \
    {                                                                                                                  \
        uart_driver_rx_irq_handler(&uart_##idx##_resources);                                                           \
    }                                                                                                                  \
                                                                                                                       \
    void USART##idx##_TX_IRQHandler(void)                                                                              \
    {                                                                                                                  \
        uart_driver_tx_irq_handler(&uart_##idx##_resources);                                                           \
    }                                                                                                                  \
                                                                                                                       \
    ARM_DRIVER_USART Driver_UART_##idx = {                                                                             \
        .GetVersion      = uart_driver_get_version,                                                                    \
        .GetCapabilities = uart_driver_get_capabilities,                                                               \
        .Initialize      = _uart_##idx##_driver_initialize,                                                            \
        .Uninitialize    = _uart_##idx##_driver_uninitialize,                                                          \
        .PowerControl    = _uart_##idx##_driver_power_control,                                                         \
        .Send            = _uart_##idx##_driver_send,                                                                  \
        .Receive         = _uart_##idx##_driver_receive,                                                               \
        .Transfer        = _uart_##idx##_driver_transfer,                                                              \
        .GetTxCount      = _uart_##idx##_driver_get_tx_count,                                                          \
        .GetRxCount      = _uart_##idx##_driver_get_rx_count,                                                          \
        .Control         = _uart_##idx##_driver_control,                                                               \
        .GetStatus       = _uart_##idx##_driver_get_status,                                                            \
        .SetModemControl = _uart_##idx##_driver_set_modem_control,                                                     \
        .GetModemStatus  = _uart_##idx##_driver_get_modem_status,                                                      \
    };                                                                                                                 \
                                                                                                                       \
    uart_pins_t uart##idx##_pins = {                                                                                   \
        .uart_rx = &GPIO_UART##idx##_RX,                                                                               \
        .rx_loc  = UART##idx##_RX_LOC,                                                                                 \
        .uart_tx = &GPIO_UART##idx##_TX,                                                                               \
        .tx_loc  = UART##idx##_TX_LOC,                                                                                 \
    };                                                                                                                 \
                                                                                                                       \
    uart_resources_t uart_##idx##_resources = {                                                                        \
        ._state      = USART_DRIVER_STATE_UNINITIALIZED,                                                               \
        ._conf       = &uart##idx##_conf,                                                                              \
        ._clock      = cmuClock_USART##idx,                                                                            \
        ._peripheral = USART##idx,                                                                                     \
        ._pins       = &uart##idx##_pins,                                                                              \
        ._rx_irq     = USART##idx##_RX_IRQn,                                                                           \
        ._tx_irq     = USART##idx##_TX_IRQn,                                                                           \
    }

/**********************************************************************************************************************/
/* Typedefs                                                                                                           */
/**********************************************************************************************************************/

/* Enums */

/* Forward declarations */

/* Structs */

typedef struct _uart_pins {
    ARM_GPIO_PIN *uart_rx;
    uint32_t       rx_loc;
    ARM_GPIO_PIN *uart_tx;
    uint32_t       tx_loc;
} uart_pins_t;

typedef struct _uart_resources {
    ARM_USART_DRIVER_STATE  _state;
    ARM_USART_CONF *        _conf;
    CMU_Clock_TypeDef       _clock;
    uart_pins_t *           _pins;
    IRQn_Type               _rx_irq;
    IRQn_Type               _tx_irq;
    USART_TypeDef *         _peripheral;
    uint8_t *               __send_buff;
    uint32_t                __send_len;
    volatile uint32_t       __bytes_sent;
    uint8_t *               __recv_buff;
    volatile uint32_t       __recv_len;
    uint32_t                __bytes_recv;
    ARM_USART_SignalEvent_t __callback;
    ARM_USART_STATUS        __status;
    ARM_USART_MODEM_STATUS  __modem_status;
} uart_resources_t;

/**********************************************************************************************************************/
/* API declaration                                                                                                    */
/**********************************************************************************************************************/
ARM_DRIVER_VERSION     uart_driver_get_version(void);
ARM_USART_CAPABILITIES uart_driver_get_capabilities(void);
int32_t                uart_driver_initialize(uart_resources_t *resources, ARM_USART_SignalEvent_t cb_event);
int32_t                uart_driver_uninitialize(uart_resources_t *resources);
int32_t                uart_driver_power_control(uart_resources_t *resources, ARM_POWER_STATE state);
int32_t                uart_driver_send(uart_resources_t *resources, const void *data, uint32_t num);
int32_t                uart_driver_receive(uart_resources_t *resources, void *data, uint32_t num);
int32_t          uart_driver_transfer(uart_resources_t *resources, const void *data_out, void *data_in, uint32_t num);
uint32_t         uart_driver_get_tx_count(uart_resources_t *resources);
uint32_t         uart_driver_get_rx_count(uart_resources_t *resources);
int32_t          uart_driver_control(uart_resources_t *resources, uint32_t control, uint32_t arg);
ARM_USART_STATUS uart_driver_get_status(uart_resources_t *resources);
int32_t          uart_driver_set_modem_control(uart_resources_t *resources, ARM_USART_MODEM_CONTROL control);
ARM_USART_MODEM_STATUS uart_driver_get_modem_status(uart_resources_t *resources);
void                   uart_driver_tx_irq_handler(uart_resources_t *resources);
void                   uart_driver_rx_irq_handler(uart_resources_t *resources);