#pragma once

/**********************************************************************************************************************/
/* Description: SiLabs EFR32 GPIO driver header file
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* C standards */
#include <stdint.h>

/* Project dependencies */
#include <Driver_GPIO.h>
#include <em_gpio.h>

/**********************************************************************************************************************/
/* Macros                                                                                                             */
/**********************************************************************************************************************/
#define CREATE_GPIO_PIN(name, in_port, in_pin)                                                                         \
    ARM_GPIO_PIN GPIO_##name = {                                                                                       \
        .port = in_port,                                                                                               \
        .pin  = in_pin,                                                                                                \
    }

#define CREATE_GPIO_DRIVER                                                                                             \
    extern ARM_DRIVER_GPIO  Driver_GPIO;                                                                               \
    extern gpio_resources_t resources;                                                                                 \
                                                                                                                       \
    static int32_t _gpio_driver_initialize(void)                                                                       \
    {                                                                                                                  \
        return gpio_driver_initialize(&resources);                                                                     \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _gpio_driver_uninitialize(void)                                                                     \
    {                                                                                                                  \
        return gpio_driver_uninitialize(&resources);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _gpio_driver_power_control(ARM_POWER_STATE state)                                                   \
    {                                                                                                                  \
        return gpio_driver_power_control(&resources, state);                                                           \
    }                                                                                                                  \
                                                                                                                       \
    static int32_t _gpio_driver_control(ARM_GPIO_PIN *pin, uint32_t control, uint32_t arg)                             \
    {                                                                                                                  \
        return gpio_driver_control(&resources, pin, control, arg);                                                     \
    }                                                                                                                  \
                                                                                                                       \
    void GPIO_EVEN_IRQHandler(void)                                                                                    \
    {                                                                                                                  \
        gpio_driver_irq_handler(&resources, GPIO_EVEN);                                                                \
    }                                                                                                                  \
                                                                                                                       \
    void GPIO_ODD_IRQHandler(void)                                                                                     \
    {                                                                                                                  \
        gpio_driver_irq_handler(&resources, GPIO_ODD);                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    ARM_DRIVER_GPIO Driver_GPIO = {                                                                                    \
        .GetVersion      = gpio_driver_get_version,                                                                    \
        .GetCapabilities = gpio_driver_get_capabilities,                                                               \
        .Initialize      = _gpio_driver_initialize,                                                                    \
        .Uninitialize    = _gpio_driver_uninitialize,                                                                  \
        .PowerControl    = _gpio_driver_power_control,                                                                 \
        .Control         = _gpio_driver_control,                                                                       \
        .GetLevel        = gpio_driver_get_level,                                                                      \
        .SetLevel        = gpio_driver_set_level,                                                                      \
    };                                                                                                                 \
                                                                                                                       \
    gpio_resources_t resources = {                                                                                     \
        .state     = GPIO_DRIVER_STATE_UNINITIALIZED,                                                                  \
        .subs      = 0,                                                                                                \
        .callbacks = {0},                                                                                              \
    }

/**********************************************************************************************************************/
/* Typedefs                                                                                                           */
/**********************************************************************************************************************/

/* Enums */

/* Definition for GPIO pin parity enum */
typedef enum {
    GPIO_EVEN = 0,
    GPIO_ODD,
} gpio_parity_e;

/* Structs */

/* Definition for GPIO pin struct */
typedef struct _ARM_GPIO_PIN {
    GPIO_Port_TypeDef port;
    uint8_t           pin;
    GPIO_Mode_TypeDef mode;
} ARM_GPIO_PIN;

typedef struct _gpio_irq_config {
    gpio_irq_cb callback;
    uint32_t    irq_trigger;
} gpio_irq_config_t;

typedef struct _gpio_resources {
    ARM_GPIO_DRIVER_STATE state;
    uint8_t               subs;
    gpio_irq_cb           callbacks[GPIO_EXTINTNO_MAX + 1];
} gpio_resources_t;

/**********************************************************************************************************************/
/* API Declaration                                                                                                    */
/**********************************************************************************************************************/
ARM_DRIVER_VERSION    gpio_driver_get_version(void);
ARM_GPIO_CAPABILITIES gpio_driver_get_capabilities(void);
int32_t               gpio_driver_initialize(gpio_resources_t *resources);
int32_t               gpio_driver_uninitialize(gpio_resources_t *resources);
int32_t               gpio_driver_power_control(gpio_resources_t *resources, ARM_POWER_STATE state);
int32_t        gpio_driver_control(gpio_resources_t *resources, ARM_GPIO_PIN *pin, uint32_t control, uint32_t arg);
ARM_GPIO_LEVEL gpio_driver_get_level(ARM_GPIO_PIN *pin);
int32_t        gpio_driver_set_level(ARM_GPIO_PIN *pin, ARM_GPIO_LEVEL level);
void           gpio_driver_irq_handler(gpio_resources_t *resources, gpio_parity_e parity);
