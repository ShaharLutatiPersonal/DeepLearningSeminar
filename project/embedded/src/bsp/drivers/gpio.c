/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* Own header */
#include <gpio.h>

/* Project dependencies */
#include <em_cmu.h>
#include <global.h>

/**********************************************************************************************************************/
/* Macros                                                                                                             */
/**********************************************************************************************************************/
#define ARM_GPIO_DRIVER_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0) /* Driver version */

/**********************************************************************************************************************/
/* Variables                                                                                                          */
/**********************************************************************************************************************/
static ARM_GPIO_CAPABILITIES driver_capabilities = {
    .irqs      = true,
    .low_power = false,
};

static GPIO_Mode_TypeDef mode_conversion_table[] = {
    [_ARM_GPIO_PIN_MODE_DISABLED]            = gpioModeDisabled,
    [_ARM_GPIO_PIN_MODE_INPUT]               = gpioModeInput,
    [_ARM_GPIO_PIN_MODE_INPUT_PULL]          = gpioModeInputPull,
    [_ARM_GPIO_PIN_MODE_PUSH_PULL]           = gpioModePushPull,
    [_ARM_GPIO_PIN_MODE_PUSH_PULL_ALTERNATE] = gpioModePushPullAlternate,
    [_ARM_GPIO_PIN_MODE_OPEN_DRAIN]          = gpioModeWiredOr,
};

static ARM_GPIO_LEVEL level_conversion_table[] = {
    [_ARM_GPIO_PIN_LEVEL_LOW]  = LOW,
    [_ARM_GPIO_PIN_LEVEL_HIGH] = HIGH,
};

/**********************************************************************************************************************/
/* Private functions declaration                                                                                      */
/**********************************************************************************************************************/
static GPIO_Mode_TypeDef parse_pin_mode(uint32_t arg);
static ARM_GPIO_LEVEL    parse_pin_level(uint32_t arg);

/**********************************************************************************************************************/
/* API Implementation                                                                                                 */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Description: Returns driver version */
/* Inputs     : none */
/* Outputs    : ARM_DRIVER_VERSION - driver version */
/**********************************************************************************************************************/
ARM_DRIVER_VERSION gpio_driver_get_version(void)
{
    return (ARM_DRIVER_VERSION){
        .api = ARM_GPIO_API_VERSION,
        .drv = ARM_GPIO_DRIVER_VERSION,
    };
}

/**********************************************************************************************************************/
/* Description: Returns driver capabilities */
/* Inputs     : none */
/* Outputs    : ARM_GPIO_CAPABILITIES - driver capabilities */
/**********************************************************************************************************************/
ARM_GPIO_CAPABILITIES gpio_driver_get_capabilities(void)
{
    return driver_capabilities;
}

/**********************************************************************************************************************/
/* Description: Initialize GPIO peripheral and driver */
/* Inputs     : resources - pointer to driver resources */
/* Outputs    : int32 - execution status */
/**********************************************************************************************************************/
int32_t gpio_driver_initialize(gpio_resources_t *resources)
{
    assert(resources != NULL);

    if (resources->subs == 0)
    {
        CMU_ClockEnable(cmuClock_GPIO, true);

        /* Set IRQs */
        NVIC_ClearPendingIRQ(GPIO_EVEN_IRQn);
        NVIC_ClearPendingIRQ(GPIO_ODD_IRQn);

        NVIC_EnableIRQ(GPIO_EVEN_IRQn);
        NVIC_EnableIRQ(GPIO_ODD_IRQn);
        resources->state = GPIO_DRIVER_STATE_INITIALIZED;
    }

    resources->subs++;
    return ARM_DRIVER_OK;
}

/**********************************************************************************************************************/
/* Description: Uninitialize GPIO peripheral and driver */
/* Inputs     : resources - pointer to driver resources */
/* Outputs    : int32_t - execution status */
/**********************************************************************************************************************/
int32_t gpio_driver_uninitialize(gpio_resources_t *resources)
{
    assert(resources != NULL);
    assert(resources->state != GPIO_DRIVER_STATE_POWERED); /* Uninitialize only stopped driver */

    if (resources->state == GPIO_DRIVER_STATE_UNINITIALIZED)
    {
        return ARM_DRIVER_OK;
    }

    resources->subs--;

    /* All subsribers uninitialized */
    if (resources->subs == 0)
    {
        resources->state = GPIO_DRIVER_STATE_UNINITIALIZED;

        NVIC_DisableIRQ(GPIO_ODD_IRQn);
        NVIC_DisableIRQ(GPIO_EVEN_IRQn);

        NVIC_ClearPendingIRQ(GPIO_ODD_IRQn);
        NVIC_ClearPendingIRQ(GPIO_EVEN_IRQn);

        CMU_ClockEnable(cmuClock_GPIO, false);
    }

    return ARM_DRIVER_OK;
}

/**********************************************************************************************************************/
/* Description: Control GPIO driver power */
/* Inputs     : resources - pointer to driver resources */
/*              state     - desired power mode */
/* Outputs    : int32_t  - execution status */
/**********************************************************************************************************************/
int32_t gpio_driver_power_control(gpio_resources_t *resources, ARM_POWER_STATE state)
{
    assert(resources != NULL);
    assert(resources->state != GPIO_DRIVER_STATE_UNINITIALIZED);

    switch (state)
    {
        case ARM_POWER_OFF:
        {
            resources->state = GPIO_DRIVER_STATE_INITIALIZED;
            break;
        }
        case ARM_POWER_LOW:
        {
            return ARM_DRIVER_ERROR_UNSUPPORTED;
        }
        case ARM_POWER_FULL:
        {
            resources->state = GPIO_DRIVER_STATE_POWERED;
            break;
        }
        default:
        {
            return ARM_DRIVER_ERROR_PARAMETER;
        }
    }

    return ARM_DRIVER_OK;
}

/**********************************************************************************************************************/
/* Description: Control GPIO driver and pins */
/* Inputs     : resources - pointer to driver resources */
/*              pin - GPIO pin to control */
/*              control - control code */
/*              arg - additional optional argument */
/* Outputs    : int32_t - execution status */
/**********************************************************************************************************************/
int32_t gpio_driver_control(gpio_resources_t *resources, ARM_GPIO_PIN *pin, uint32_t control, uint32_t arg)
{
    assert(resources != NULL);
    assert(resources->state == GPIO_DRIVER_STATE_POWERED);

    /* Pins controls */
    switch (control & ARM_GPIO_PIN_CONTROL_Msk)
    {
        case ARM_GPIO_SET_PIN_MODE:
        {
            if (!IS_GPIO_CONNECTED(pin))
             {return ARM_DRIVER_OK;}
            GPIO_Mode_TypeDef mode  = parse_pin_mode(arg);
            ARM_GPIO_LEVEL    level = parse_pin_level(arg);
            GPIO_PinModeSet(pin->port, pin->pin, mode, level);
            pin->mode = mode;
            break;
        }
        case ARM_GPIO_SET_PIN_STRENGTH:
        {
            return ARM_DRIVER_ERROR_UNSUPPORTED;
        }
        case ARM_GPIO_SET_PIN_IRQ_TRIGGER:
        {
            if (!IS_GPIO_CONNECTED(pin))
             {return ARM_DRIVER_OK;}
            assert(arg != 0);
            if (resources->callbacks[pin->pin] != NULL)
            {
                return ARM_GPIO_ERROR_IRQ_REASSIGN;
            }

            gpio_irq_config_t *config      = (gpio_irq_config_t *)arg;
            resources->callbacks[pin->pin] = config->callback;

            bool rising  = (config->irq_trigger & ARM_GPIO_IRQ_TRIGGER_EDGE_RISING);
            bool falling = (config->irq_trigger & ARM_GPIO_IRQ_TRIGGER_EDGE_FALLING);

            GPIO_IntConfig(pin->port, pin->pin, rising, falling, true);
            uint32_t flag = (1 << pin->pin);
            GPIO_IntEnable(flag);
            break;
        }
        default:
        {
            break;
        }
    }

    return ARM_DRIVER_OK;
}

/**********************************************************************************************************************/
/* Description: Get pin level */
/* Inputs     : pin - GPIO pin to read */
/* Outputs    : ARM_GPIO_LEVEL - read logic level */
/**********************************************************************************************************************/
ARM_GPIO_LEVEL gpio_driver_get_level(ARM_GPIO_PIN *pin)
{
    assert(pin != NULL);
    return (ARM_GPIO_LEVEL)GPIO_PinOutGet(pin->port, pin->pin);
}

/**********************************************************************************************************************/
/* Description: Set pin level */
/* Inputs     : pin - GPIO pin to set */
/*              level - desired logic level */
/* Outputs    : int32_t - execution status */
/**********************************************************************************************************************/
int32_t gpio_driver_set_level(ARM_GPIO_PIN *pin, ARM_GPIO_LEVEL level)
{
    assert(pin != NULL);
    if (level == LOW)
    {
        GPIO_PinOutClear(pin->port, pin->pin);
    } else
    {
        GPIO_PinOutSet(pin->port, pin->pin);
    }

    return ARM_DRIVER_OK;
}

/**********************************************************************************************************************/
/* Description: IRQ handler for GPIO driver */
/* Inputs     : parity - parity of GPIO IRQ */
/* Outputs    : none */
/**********************************************************************************************************************/
void gpio_driver_irq_handler(gpio_resources_t *resources, gpio_parity_e parity)
{
    uint32_t irq_flags = GPIO_IntGet();
    for (uint8_t i = 0; i <= GPIO_EXTINTNO_MAX; i++)
    {
        uint32_t bit = (1 << i);
        if (irq_flags & bit)
        {
            GPIO_IntClear(bit);
            if (resources->callbacks[i] != NULL)
            {
                resources->callbacks[i]();
            }
        }
    }
}

/**********************************************************************************************************************/
/* Private functions Implementation                                                                                   */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Description: Parse GPIO mode */
/* Inputs     : arg - user argument for GPIO mode */
/* Outputs    : GPIO_Mode_TypeDef - parsed mode */
/**********************************************************************************************************************/
static GPIO_Mode_TypeDef parse_pin_mode(uint32_t arg)
{
    uint16_t pin_mode = (arg & ARM_GPIO_PIN_MODE_Msk) >> ARM_GPIO_PIN_MODE_Pos;
    assert(pin_mode > 0 && pin_mode < 0xf);
    return mode_conversion_table[pin_mode];
}

/**********************************************************************************************************************/
/* Description: Parse GPIO level */
/* Inputs     : arg - user argument for GPIO level */
/* Outputs    : ARM_GPIO_LEVEL - parsed level */
/**********************************************************************************************************************/
static ARM_GPIO_LEVEL parse_pin_level(uint32_t arg)
{
    uint8_t pin_level = (arg & ARM_GPIO_PIN_LEVEL_Msk) >> ARM_GPIO_PIN_LEVEL_Pos;
    assert(pin_level > 0 || pin_level < 3);
    return level_conversion_table[pin_level];
}