#pragma once
/**********************************************************************************************************************/
/* Description: Silabs EFR32 I2S driver header */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* C Standards */
#include <stdint.h>

/* Project dependencies */
#include <Driver_GPIO.h>
#include <Driver_SAI.h>
#include <em_cmu.h>
#include <em_gpio.h>
#include <em_usart.h>
#include <ldma.h>

/**********************************************************************************************************************/
/* Macros                                                                                                             */
/**********************************************************************************************************************/
#define CREATE_I2S_DRIVER(idx)                                                                                         \
    extern ARM_DRIVER_SAI  Driver_I2S_##idx;                                                                           \
    extern i2s_resources_t i2s_##idx##_resources;                                                                      \
                                                                                                                       \
    static int32_t _i2s_##idx##_driver_initialize(ARM_SAI_SignalEvent_t cb_event)                                      \
    {                                                                                                                  \
        return i2s_driver_initialize(&i2s_##idx##_resources, cb_event);                                                \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _i2s_##idx##_driver_uninitialize(void)                                                              \
    {                                                                                                                  \
        return i2s_driver_uninitialize(&i2s_##idx##_resources);                                                        \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _i2s_##idx##_driver_power_control(ARM_POWER_STATE state)                                            \
    {                                                                                                                  \
        return i2s_driver_power_control(&i2s_##idx##_resources, state);                                                \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _i2s_##idx##_driver_send(const void *data, uint32_t num)                                            \
    {                                                                                                                  \
        return i2s_driver_send(&i2s_##idx##_resources, data, num);                                                     \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _i2s_##idx##_driver_receive(void *data, uint32_t num)                                               \
    {                                                                                                                  \
        return i2s_driver_receive(&i2s_##idx##_resources, data, num);                                                  \
    }                                                                                                                  \
                                                                                                                       \
    static uint32_t _i2s_##idx##_driver_get_tx_count(void)                                                             \
    {                                                                                                                  \
        return i2s_driver_get_tx_count(&i2s_##idx##_resources);                                                        \
    }                                                                                                                  \
                                                                                                                       \
    static uint32_t _i2s_##idx##_driver_get_rx_count(void)                                                             \
    {                                                                                                                  \
        return i2s_driver_get_rx_count(&i2s_##idx##_resources);                                                        \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _i2s_##idx##_driver_control(uint32_t control, uint32_t arg)                                         \
    {                                                                                                                  \
        return i2s_driver_control(&i2s_##idx##_resources, control, arg);                                               \
    }                                                                                                                  \
                                                                                                                       \
    static ARM_SAI_STATUS _i2s_##idx##_driver_get_status(void)                                                         \
    {                                                                                                                  \
        return i2s_driver_get_status(&i2s_##idx##_resources);                                                          \
    }                                                                                                                  \
                                                                                                                       \
    ARM_DRIVER_SAI Driver_SAI_##idx = {                                                                                \
        .GetVersion      = i2s_driver_get_version,                                                                     \
        .GetCapabilities = i2s_driver_get_capabilities,                                                                \
        .Initialize      = _i2s_##idx##_driver_initialize,                                                             \
        .Uninitialize    = _i2s_##idx##_driver_uninitialize,                                                           \
        .PowerControl    = _i2s_##idx##_driver_power_control,                                                          \
        .Send            = _i2s_##idx##_driver_send,                                                                   \
        .Receive         = _i2s_##idx##_driver_receive,                                                                \
        .GetTxCount      = _i2s_##idx##_driver_get_tx_count,                                                           \
        .GetRxCount      = _i2s_##idx##_driver_get_rx_count,                                                           \
        .Control         = _i2s_##idx##_driver_control,                                                                \
        .GetStatus       = _i2s_##idx##_driver_get_status,                                                             \
    };                                                                                                                 \
                                                                                                                       \
    i2s_pins_t i2s##idx##_pins = {                                                                                     \
        .i2s_clk = &GPIO_I2S##idx##_CLK,                                                                               \
        .clk_loc = I2S##idx##_CLK_LOC,                                                                                 \
        .i2s_ws  = &GPIO_I2S##idx##_WS,                                                                                \
        .ws_loc  = I2S##idx##_WS_LOC,                                                                                  \
        .i2s_rx  = &GPIO_I2S##idx##_RX,                                                                                \
        .rx_loc  = I2S##idx##_RX_LOC,                                                                                  \
        .i2s_tx  = &GPIO_I2S##idx##_TX,                                                                                \
        .tx_loc  = I2S##idx##_TX_LOC,                                                                                  \
    };                                                                                                                 \
                                                                                                                       \
    i2s_resources_t i2s_##idx##_resources = {                                                                          \
        ._state      = SAI_DRIVER_STATE_UNINITIALIZED,                                                                 \
        ._conf       = &i2s##idx##_conf,                                                                               \
        ._clock      = cmuClock_USART##idx,                                                                            \
        ._peripheral = USART##idx,                                                                                     \
        ._pins       = &i2s##idx##_pins,                                                                               \
        ._rx_irq     = USART##idx##_RX_IRQn,                                                                           \
        ._tx_irq     = USART##idx##_TX_IRQn,                                                                           \
    }

/**********************************************************************************************************************/
/* Typedefs                                                                                                           */
/**********************************************************************************************************************/

/* Enums */

/* Forward declarations */

/* Structs */

typedef struct _i2s_pins {
    ARM_GPIO_PIN *i2s_clk;
    uint8_t       clk_loc;
    ARM_GPIO_PIN *i2s_ws;
    uint8_t       ws_loc;
    ARM_GPIO_PIN *i2s_rx;
    uint8_t       rx_loc;
    ARM_GPIO_PIN *i2s_tx;
    uint8_t       tx_loc;
} i2s_pins_t;

typedef struct _i2s_resources {
    ARM_SAI_DRIVER_STATE  _state;
    ARM_SAI_CONF *        _conf;
    CMU_Clock_TypeDef     _clock;
    i2s_pins_t *          _pins;
    USART_TypeDef *       _peripheral;
    uint8_t *             __send_buff;
    uint32_t              __send_len;
    uint8_t *             __send_buff_alt;
    uint32_t              __send_len_alt;
    volatile uint32_t     __bytes_sent;
    uint8_t *             __recv_buff;
    volatile uint32_t     __recv_len;
    uint8_t *             __recv_buff_alt;
    volatile uint32_t     __recv_len_alt;
    uint32_t              __bytes_recv;
    ARM_SAI_SignalEvent_t __callback;
    ARM_SAI_STATUS        __status;
} i2s_resources_t;

/**********************************************************************************************************************/
/* API declaration                                                                                                    */
/**********************************************************************************************************************/
ARM_DRIVER_VERSION   i2s_driver_get_version(void);
ARM_SAI_CAPABILITIES i2s_driver_get_capabilities(void);
int32_t              i2s_driver_initialize(i2s_resources_t *resources, ARM_SAI_SignalEvent_t cb_event);
int32_t              i2s_driver_uninitialize(i2s_resources_t *resources);
int32_t              i2s_driver_power_control(i2s_resources_t *resources, ARM_POWER_STATE state);
int32_t              i2s_driver_send(i2s_resources_t *resources, const void *data, uint32_t num);
int32_t              i2s_driver_receive(i2s_resources_t *resources, void *data, uint32_t num);
uint32_t             i2s_driver_get_tx_count(i2s_resources_t *resources);
uint32_t             i2s_driver_get_rx_count(i2s_resources_t *resources);
int32_t              i2s_driver_control(i2s_resources_t *resources, uint32_t control, uint32_t arg);
ARM_SAI_STATUS       i2s_driver_get_status(i2s_resources_t *resources);
